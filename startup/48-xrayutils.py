print(f'Loading {__file__}...')


import xraylib
import skbeam.core.constants.xrf as xrfC


# def setroi(element=None, line='Ka', roisize=200, roi=1):
#     '''
#     setting rois for Xspress3 by providing elements
#     input:
#         element (string): element of interest, e.g. 'Se'
#         line (string): default 'Ka', options:'Ka', 'Kb', 'La', 'Lb', 'M', Ka1',
#                                              'Ka2', 'Kb1', 'Kb2', 'Lb2', 'Ma1'
#         roisize (int): roi window size in the unit of eV, default = 200
#         roi (int): 1, 2, or 3: the roi number to set, default = 1
#     '''

#     atomic_num = xraylib.SymbolToAtomicNumber(element)
#     multilineroi_flag = False
#     print('Element: ', element)
#     print('ROI window size (eV): ', roisize)

#     # Calculate the roi low and high bound, based on the line
#     # Use average of Ka1 and Ka2 as the center of the energy/roi
#     if line == 'Ka':
#         line_h = xraylib.LineEnergy(atomic_num, xraylib.KL3_LINE) * 1000  # Ka1
#         line_l = xraylib.LineEnergy(atomic_num, xraylib.KL2_LINE) * 1000  # Ka2
#         print('Ka1 line (eV): ', line_h)
#         print('Ka2 line (eV): ', line_l)
#         energy_cen = (line_l + line_h) / 2
#         multilineroi_flag = True
#     # Using Kb1 line only as the center of the energy/roi
#     elif line == 'Kb':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KM3_LINE) * 1000
#         print('Kb1 line (eV): ', energy_cen)
#     # Using average of La1 and La2 as the center of the energy/roi
#     elif line == 'La':
#         line_h = xraylib.LineEnergy(atomic_num, xraylib.L3M5_LINE) * 1000  # La1
#         line_l = xraylib.LineEnergy(atomic_num, xraylib.L3M4_LINE) * 1000  # La2
#         print('La1 line (eV): ', line_h)
#         print('La2 line (eV): ', line_l)
#         energy_cen = (line_l + line_h) / 2
#         multilineroi_flag = True
#     # Using average of Lb1 and Lb2 as the center of the energy/roi
#     elif line == 'Lb':
#         line_l = xraylib.LineEnergy(atomic_num, xraylib.L2M4_LINE) * 1000  # Lb1
#         line_h = xraylib.LineEnergy(atomic_num, xraylib.L3N5_LINE) * 1000  # Lb2
#         print('Lb2 line (eV): ', line_h)
#         print('Lb1 line (eV): ', line_l)
#         energy_cen = (line_l + line_h) / 2
#         multilineroi_flag = True
#     # Using Ma1 line only as the center of the energy/roi
#     elif line == 'M':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.M5N7_LINE) * 1000
#         print('Ma1 line (eV): ', energy_cen)
#     elif line == 'Ka1':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KL3_LINE) * 1000
#         print('Ka1 line (eV): ', energy_cen)
#     elif line == 'Ka2':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KL2_LINE) * 1000
#         print('Ka2 line (eV): ', energy_cen)
#     elif line == 'Kb1':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KM3_LINE) * 1000
#         print('Kb1 line (eV): ', energy_cen)
#     elif line == 'Lb1':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.L2M4_LINE) * 1000
#         print('Kb2 line (eV): ', energy_cen)
#     elif line == 'Lb2':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.L3N5_LINE) * 1000
#         print('Lb2 line (eV): ', energy_cen)
#     elif line == 'Ma1':
#         energy_cen = xraylib.LineEnergy(atomic_num, xraylib.M5N7_LINE) * 1000
#         print('Ma1 line (eV): ', energy_cen)

#     print('Energy center (eV): ', energy_cen)

#     # Converting energy center position from keV to eV, then to channel number
#     roi_cen = energy_cen / 10.
#     roi_l = round(roi_cen - roisize / 10 / 2)
#     roi_h = round(roi_cen + roisize / 10 / 2)

#     print('ROI center: ', roi_cen)
#     print('ROI lower bound:', roi_l, ' (', roi_l * 10, ' eV)')
#     print('ROI higher bound: ', roi_h, ' (', roi_h * 10, ' eV)')

#     if roi_l <= 0:
#         raise Exception('Lower roi bound is at or less than zero.')
#     if roi_h >= 2048:
#         raise Exception('Higher roi bound is at or larger than 2048.')

#     if multilineroi_flag is True:
#         print(f"Lowest emission line to roi lower bound: "
#               "{line_l - roi_l * 10} eV")
#         print(f"Highest emission line to roi higher bound: "
#               "{line_h - roi_h * 10} eV")

#         if roi_l*10 - line_l > 0:
#             print(f"Warning: Window does not cover the lower emission line."
#                   "Consider making roisize larger.\n"
#                   "Currently the window lower bound is higher than lower "
#                   "emission line by {roi_l*10 - line_l} eV")
#         if line_h - roi_h*10 > 0:
#             print(f"Warning: Window does not cover the higher emission line."
#                   "Consider making roisize larger.\n"
#                   "Currently the window higher bound is less than higher "
#                   "emission line by {line_h - roi_h * 10} eV")

#     # set up roi values
#     if roi == 1:
#         xs.channel1.rois.roi01.bin_low.set(roi_l)
#         xs.channel1.rois.roi01.bin_high.set(roi_h)
#         xs.channel2.rois.roi01.bin_low.set(roi_l)
#         xs.channel2.rois.roi01.bin_high.set(roi_h)
#         xs.channel3.rois.roi01.bin_low.set(roi_l)
#         xs.channel3.rois.roi01.bin_high.set(roi_h)
#     elif roi == 2:
#         xs.channel1.rois.roi02.bin_low.set(roi_l)
#         xs.channel1.rois.roi02.bin_high.set(roi_h)
#         xs.channel2.rois.roi02.bin_low.set(roi_l)
#         xs.channel2.rois.roi02.bin_high.set(roi_h)
#         xs.channel3.rois.roi02.bin_low.set(roi_l)
#         xs.channel3.rois.roi02.bin_high.set(roi_h)
#     elif roi == 3:
#         xs.channel1.rois.roi03.bin_low.set(roi_l)
#         xs.channel1.rois.roi03.bin_high.set(roi_h)
#         xs.channel2.rois.roi03.bin_low.set(roi_l)
#         xs.channel2.rois.roi03.bin_high.set(roi_h)
#         xs.channel3.rois.roi03.bin_low.set(roi_l)
#         xs.channel3.rois.roi03.bin_high.set(roi_h)
#     else:
#         print('Cannot set ROI values; ROI = 1, 2, or 3')


def edge(element=None, line='K', unit='eV'):
    '''
    function return edge (K or L3) in eV or keV with input element sympbol
    '''

    atomic_num = xraylib.SymbolToAtomicNumber(element)

    if line == 'K':
        edge_value = xraylib.EdgeEnergy(atomic_num, xraylib.K_SHELL)
    if line == 'L3':
        edge_value = xraylib.EdgeEnergy(atomic_num, xraylib.L3_SHELL)        

    if unit == 'eV':
        return edge_value * 1000
    else:
        return edge_value


interestinglist = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                   'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                   'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                   'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                   'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                   'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                   'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                   'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

elements = dict()
element_edges = ['ka1', 'ka2', 'kb1', 'la1', 'la2', 'lb1', 'lb2', 'lg1', 'ma1']
element_transitions = ['k', 'l1', 'l2', 'l3', 'm1', 'm2', 'm3', 'm4', 'm5']
for i in interestinglist:
    elements[i] = xrfC.XrfElement(i)


def setroi(roinum, element, edge=None, det=None):
    '''
    Set energy ROIs for Vortex SDD.
    Selects elemental edge given current energy if not provided.
    roinum      [1,2,3]     ROI number
    element     <symbol>    element symbol for target energy
    edge                    optional:  ['ka1', 'ka2', 'kb1', 'la1', 'la2',
                                        'lb1', 'lb2', 'lg1', 'ma1']
    '''
    cur_element = xrfC.XrfElement(element)
    if edge is None:
        for e in ['ka1', 'ka2', 'kb1', 'la1', 'la2',
                  'lb1', 'lb2', 'lg1', 'ma1']:
            if cur_element.emission_line[e] < energy.energy.get()[1]:
                edge = 'e'
                break
    else:
        e = edge

    e_ch = int(cur_element.emission_line[e] * 1000)
    if det is not None:
        det.channel1.set_roi(roinum, e_ch-100, e_ch+100,
                             name=element + '_' + e)
        cpt = getattr(det.channel1.rois, f'roi{roinum:02d}')
        cpt.kind = 'hinted'
    else:
        for d in [xs.channel1, xs.channel2, xs.channel3]:
            d.set_roi(roinum, e_ch-100, e_ch+100, name=element + '_' + e)
            cpt = getattr(d.rois, f'roi{roinum:02d}')
            cpt.kind = 'hinted'
    print("ROI{} set for {}-{} edge.".format(roinum, element, e))


def clearroi(roinum=None):
    if roinum is None:
        roinum = [1, 2, 3]
    try:
        roinum = list(roinum)
    except TypeError:
        roinum = [roinum]

    # xs.channel1.rois.roi01.clear
    for d in [xs.channel1.rois, xs.channel2.rois, xs.channel3.rois]:
        for roi in roinum:
            cpt = getattr(d, f'roi{roi:02d}')
            cpt.clear()
            cpt.kind = 'omitted'


def getemissionE(element, edge=None):
    cur_element = xrfC.XrfElement(element)
    if edge is None:
        print("Edge\tEnergy [keV]")
        for e in element_edges:
            if cur_element.emission_line[e] < 25. and \
               cur_element.emission_line[e] > 1.:
                # print("{0:s}\t{1:8.2f}".format(e, cur_element.emission_line[e]))
                print(f"{e}\t{cur_element.emission_line[e]:8.2f}")
    else:
        return cur_element.emission_line[edge]


def getbindingE(element, edge=None):
    '''
    Return edge energy in eV if edge is specified,
    otherwise return K and L edge energies and yields
    element     <symbol>        element symbol for target
    edge        ['k','l1','l2','l3']    return binding energy of this edge
    '''
    if edge is None:
        y = [0., 'k']
        print("Edge\tEnergy [eV]\tYield")
        for i in ['k', 'l1', 'l2', 'l3']:
            # print("{0:s}\t{1:8.2f}\t{2:5.3}".format(i,xrfC.XrayLibWrap(elements[element].Z,'binding_e')[i]*1000.,
            #                                       xrfC.XrayLibWrap(elements[element].Z,'yield')[i]))
            print(f"{i}\t"
                  f"{xrfC.XrayLibWrap(elements[element].Z,'binding_e')[i]*1000.:8.2f}\t"
                  f"{xrfC.XrayLibWrap(elements[element].Z,'yield')[i]:5.3f}")
            if (y[0] < xrfC.XrayLibWrap(elements[element].Z, 'yield')[i] and
               xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[i] < 25.):
                y[0] = xrfC.XrayLibWrap(elements[element].Z, 'yield')[i]
                y[1] = i
        return xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[y[1]] * 1000.
    else:
        return xrfC.XrayLibWrap(elements[element].Z, 'binding_e')[edge] * 1000.
