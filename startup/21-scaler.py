
class EpicsSignalROLazyier(EpicsSignalRO):
    def get(self, *args, timeout=5, **kwargs):
        return super().get(*args, timeout=timeout, **kwargs)

def _scaler_fields(attr_base, field_base, range_, **kwargs):
    defn = OrderedDict()
    for i in range_:
        attr = '{attr}{i}'.format(attr=attr_base, i=i)
        suffix = '{field}{i}'.format(field=field_base, i=i)
        defn[attr] = (EpicsSignalROLazyier, suffix, kwargs)

    return defn


class SRXScaler(EpicsScaler):
    acquire_mode = Cpt(EpicsSignal, 'AcquireMode')
    acquiring = Cpt(EpicsSignal, 'Acquiring')
    asyn = Cpt(EpicsSignal, 'Asyn')
    channel1_source = Cpt(EpicsSignal, 'Channel1Source')
    channel_advance = Cpt(EpicsSignal, 'ChannelAdvance', string=True)
    channels = DDC(_scaler_fields('chan', '.S', range(1, 33)))
    client_wait = Cpt(EpicsSignal, 'ClientWait')
    count_on_start = Cpt(EpicsSignal, 'CountOnStart')
    current_channel = Cpt(EpicsSignal, 'CurrentChannel')
    disable_auto_count = Cpt(EpicsSignal, 'DisableAutoCount')
    do_read_all = Cpt(EpicsSignal, 'DoReadAll')
    dwell = Cpt(EpicsSignal, 'Dwell')
    elapsed_real = Cpt(EpicsSignal, 'ElapsedReal')
    enable_client_wait = Cpt(EpicsSignal, 'EnableClientWait')
    erase_all = Cpt(EpicsSignal, 'EraseAll')
    erase_start = Cpt(EpicsSignal, 'EraseStart')
    firmware = Cpt(EpicsSignal, 'Firmware')
    hardware_acquiring = Cpt(EpicsSignal, 'HardwareAcquiring')
    input_mode = Cpt(EpicsSignal, 'InputMode')
    max_channels = Cpt(EpicsSignal, 'MaxChannels')
    model = Cpt(EpicsSignal, 'Model')
    mux_output = Cpt(EpicsSignal, 'MUXOutput')
    nuse_all = Cpt(EpicsSignal, 'NuseAll')
    output_mode = Cpt(EpicsSignal, 'OutputMode')
    output_polarity = Cpt(EpicsSignal, 'OutputPolarity')
    prescale = Cpt(EpicsSignal, 'Prescale')
    preset_real = Cpt(EpicsSignal, 'PresetReal')
    read_all = Cpt(EpicsSignal, 'ReadAll')
    read_all_once = Cpt(EpicsSignal, 'ReadAllOnce')
    set_acquiring = Cpt(EpicsSignal, 'SetAcquiring')
    set_client_wait = Cpt(EpicsSignal, 'SetClientWait')
    snl_connected = Cpt(EpicsSignal, 'SNL_Connected')
    software_channel_advance = Cpt(EpicsSignal, 'SoftwareChannelAdvance')
    count_mode=Cpt(EpicsSignal, '.CONT')
    start_all = Cpt(EpicsSignal, 'StartAll')
    stop_all = Cpt(EpicsSignal, 'StopAll')
    user_led = Cpt(EpicsSignal, 'UserLED')
    wfrm = Cpt(EpicsSignal, 'Wfrm')
    mca1 = Cpt(EpicsSignalRO, 'mca1')
    mca2 = Cpt(EpicsSignalRO, 'mca2')
    mca3 = Cpt(EpicsSignalRO, 'mca3')
    mca4 = Cpt(EpicsSignalRO, 'mca4')

    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)
        self.stage_sigs[self.count_mode] = 'OneShot'

sclr1 = SRXScaler('XF:05IDD-ES:1{Sclr:1}',name='sclr1')
sclr1.read_attrs = ['channels.chan2','channels.chan3','channels.chan4']
i0_channel = getattr(sclr1.channels,'chan2')
i0_channel.name = 'sclr_i0'
it_channel = getattr(sclr1.channels,'chan4')
it_channel.name = 'sclr_it'
im_channel = getattr(sclr1.channels,'chan3')
im_channel.name = 'sclr_im'
i0 = sclr1.channels.chan2
it = sclr1.channels.chan4
im = sclr1.channels.chan3


