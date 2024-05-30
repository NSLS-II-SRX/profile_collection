from bluesky.utils import new_uid, short_uid, make_decorator
from bluesky.preprocessors import single_gen, ensure_generator, plan_mutator
from bluesky.run_engine import IllegalMessageSequence, Msg
from bluesky.bundlers import _rearrange_into_parallel_dicts
from event_model import DocumentNames
import time as ttime


async def ts_monitor(self, msg):
    """
    Monitor a time series. Emit event pages asynchronously.

    A descriptor document is emitted immediately. Then, a closure is
    defined that emits Event documents associated with that descriptor
    from a separate thread. This process is not related to the main
    bundling process (create/read/save).

    Expected message object is::

        Msg('monitor', obj, **kwargs)
        Msg('monitor', obj, name='event-stream-name', **kwargs)

    where kwargs are passed through to ``obj.subscribe()``
    """
    obj = msg.obj
    # TODO inject this through the msg
    reset_count_name = "reset_count"
    index_name = "index_count"
    if msg.args:
        raise ValueError("The 'monitor' Msg does not accept positional " "arguments.")
    kwargs = dict(msg.kwargs)
    name = kwargs.pop("name", short_uid("monitor"))
    if obj in self._monitor_params:
        raise IllegalMessageSequence(
            "A 'monitor' message was sent for {}" "which is already monitored".format(obj)
        )
    descriptor_uid = new_uid()
    data_keys = obj.describe()
    # HACK change time series into a scalar
    for k, v in data_keys.items():
        data_keys[k] = dict(v)
        data_keys[k]["shape"] = []
        data_keys[k]["dtype"] = "number"

    if reset_count_name in data_keys:
        raise ValueError(f"A Signal name collides with a implicit name in the monitoring ({reset_count_name!r})")

    if index_name in data_keys:
        raise ValueError(f"A Signal name collides with a implicit name in the monitoring ({index_name!r})")

    data_keys[reset_count_name] = {
        "dtype": "integer",
        "shape": [],
        "source": "monitor_code",
    }

    data_keys[index_name] = {
        "dtype": "integer",
        "shape": [],
        "source": "monitor_code",
    }

    config = {obj.name: {"data": {}, "timestamps": {}}}
    config[obj.name]["data_keys"] = obj.describe_configuration()
    for key, val in obj.read_configuration().items():
        config[obj.name]["data"][key] = val["value"]
        config[obj.name]["timestamps"][key] = val["timestamp"]
    object_keys = {obj.name: list(data_keys)}
    hints = {}
    if hasattr(obj, "hints"):
        hints.update({obj.name: obj.hints})
    desc_doc = dict(
        run_start=self._run_start_uid,
        time=ttime.time(),
        data_keys=data_keys,
        uid=descriptor_uid,
        configuration=config,
        hints=hints,
        name=name,
        object_keys=object_keys,
    )

    seq_num_count = 1
    reset_count = 0
    seen = 0

    def emit_event_page(*args, **kwargs):
        nonlocal seen
        nonlocal seq_num_count
        nonlocal reset_count
        # Ignore the inputs. Use this call as a signal to call read on the
        # object, a crude way to be sure we get all the info we need.
        data, timestamps = _rearrange_into_parallel_dicts(obj.read())
        # there should only be one key in here! get it out
        ((k, ts_data),) = data.items()
        ((k, ts_ts),) = timestamps.items()
        # if we have fewer than before, assume new row and reset
        # TODO sort out a better way to do this?
        if len(ts_data) < seen:
            seen = 0
            reset_count += 1
        # trim the data we got to only the new data
        emited_data = ts_data[seen:]
        if len(emited_data) == 0:
            return
        # construct the event page
        t_current = ttime.time()
        dt_small = 1e-4
        event_page = dict(
            descriptor=descriptor_uid,
            time=[t_current + dt_small * _ for _ in range(len(emited_data))],
            data={
                k: emited_data,
                reset_count_name: [reset_count] * len(emited_data),
                index_name: list(range(seen, len(ts_data))),
            },
            timestamps={
                k: [ts_ts] * len(emited_data),
                reset_count_name: [ts_ts] * len(emited_data),
                index_name: [ts_ts] * len(emited_data),
            },
            seq_num=list(range(seq_num_count, seq_num_count + len(emited_data))),
            uid=[new_uid() for _ in range(len(emited_data))],
        )
        # update our book keeping
        seen = len(ts_data)
        seq_num_count += len(emited_data)
        # emit the event page
        self.emit_sync(DocumentNames.event_page, event_page)

    # stash function + metadata for later to un/re subscribe if needed
    self._monitor_params[obj] = emit_event_page, kwargs
    # publish the descriptor
    await self.emit(DocumentNames.descriptor, desc_doc)
    obj.subscribe(emit_event_page, **kwargs)


def ts_monitor_during_wrapper(plan, signals):
    """
    Monitor (asynchronously read) devices during runs.

    This is a preprocessor that insert messages immediately after a run is
    opened and before it is closed.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    signals : collection
        objects that support the Signal interface

    Yields
    ------
    msg : Msg
        messages from plan with 'monitor', and 'unmontior' messages inserted

    See Also
    --------
    :func:`bluesky.plans.fly_during_wrapper`
    """
    monitor_msgs = [Msg("ts_monitor", sig, name="DONOTSAVE_" + sig.name + "_monitor") for sig in signals]
    unmonitor_msgs = [Msg("unmonitor", sig) for sig in signals]

    def insert_after_open(msg):
        if msg.command == "open_run":

            def new_gen():
                yield from ensure_generator(monitor_msgs)

            return single_gen(msg), new_gen()
        else:
            return None, None

    def insert_before_close(msg):
        if msg.command == "close_run":

            def new_gen():
                yield from ensure_generator(unmonitor_msgs)
                yield msg

            return new_gen(), None
        else:
            return None, None

    # Apply nested mutations.
    plan1 = plan_mutator(plan, insert_after_open)
    plan2 = plan_mutator(plan1, insert_before_close)
    return (yield from plan2)


ts_monitor_during_decorator = make_decorator(ts_monitor_during_wrapper)


async def _ts_monitor(msg):
    # rely on closing over RE in the name space, this is dirty, but
    self = RE
    run_key = msg.run
    try:
        current_run = self._run_bundlers[run_key]
    except KeyError as ke:
        raise IllegalMessageSequence("A 'monitor' message was sent but no run is open.") from ke
    await ts_monitor(current_run, msg)
    await self._reset_checkpoint_state_coro()


RE.register_command("ts_monitor", _ts_monitor)
