import sys
sys.path.append('/nfs/xf05id1/src/nsls2-xf-utils')
import srxslit
import srxfe
import srxbpm
import tempdev
import srxm2

#wb=srxslit.nsls2slit(tb='XF:05IDA-OP:1{Slt:1-Ax:T}',bb='XF:05IDA-OP:1{Slt:1-Ax:B}',ib='XF:05IDA-OP:1{Slt:1-Ax:I}',ob='XF:05IDA-OP:1{Slt:1-Ax:O}')
#pb=srxslit.nsls2slit(ib='XF:05IDA-OP:1{Slt:2-Ax:I}',ob='XF:05IDA-OP:1{Slt:2-Ax:O}')
#ssa=srxslit.nsls2slit(tb='XF:05IDB-OP:1{Slt:SSA-Ax:T}', bb='XF:05IDB-OP:1{Slt:SSA-Ax:B}', ob='XF:05IDB-OP:1{Slt:SSA-Ax:O}',ib='XF:05IDB-OP:1{Slt:SSA-Ax:I}')
m2x=srxm2.mottwin(m1='XF:05IDD-OP:1{Mir:2-Ax:XU}Mtr',m2='XF:05IDD-OP:1{Mir:2-Ax:XD}Mtr')
# bpm1=srxbpm.nsls2bpm(bpm='bpm1')
# bpm2=srxbpm.nsls2bpm(bpm='bpm2')
