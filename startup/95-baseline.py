# Setup baseline measurements for scans
print(f'Loading {__file__}...')

from bluesky.preprocessors import SupplementalData

sd = SupplementalData()

sd.baseline = [ring_current, fe, energy, dcm, hfm,             # Front-end slits, Undulator/Bragg, HDCM, HFM
               slt_wb, slt_pb, slt_ssa,                        # White-, Pink-Beam slits, SSA
               jjslits, attenuators,                           # JJ slits, Attenuator Box
               nanoKB, nano_vlm_stage, nano_det, temp_nanoKB,  # nanoKBs, VLM, Detector, Temperatures
               nano_stage]                                     # coarse/fine stages

RE.preprocessors.append(sd)

