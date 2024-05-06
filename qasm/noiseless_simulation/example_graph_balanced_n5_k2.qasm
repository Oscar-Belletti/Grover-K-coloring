OPENQASM 3;
include "stdgates.inc";
gate mcx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  p(pi/8) _gate_q_0;
  p(pi/8) _gate_q_1;
  p(pi/8) _gate_q_2;
  p(pi/8) _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  p(-pi/8) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  cx _gate_q_1, _gate_q_2;
  p(-pi/8) _gate_q_2;
  cx _gate_q_0, _gate_q_2;
  p(pi/8) _gate_q_2;
  cx _gate_q_1, _gate_q_2;
  p(-pi/8) _gate_q_2;
  cx _gate_q_0, _gate_q_2;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  p(pi/8) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  p(pi/8) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  p(pi/8) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  h _gate_q_3;
}
gate cu1_129723049454608(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(0, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
}
gate cu1_129722972463360(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
}
gate cu1_129722855488240(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722855485984(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722958996432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722958996192(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722958996960(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722958997536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722958997968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722855488240(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722855485984(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722958996432(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722958996192(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722958996960(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722958997536(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722958997968(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx_129722869770848 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129723049454608(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722972463360(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722651497760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722651498816(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129723064327904 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
}
gate mcx_129722950344224 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722651497760(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722651498816(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129723064327904 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722958271344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722958273024(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722958273408 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
}
gate mcx_129723051310816 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722958271344(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722958273024(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722958273408 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
bit[5] _creg;
qubit[9] q96;
h q96[0];
h q96[1];
h q96[2];
h q96[3];
h q96[4];
x q96[8];
h q96[8];
x q96[5];
x q96[6];
x q96[7];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
x q96[7];
ccx q96[5], q96[6], q96[7];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[0], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[0], q96[2];
cx q96[1], q96[3];
x q96[3];
cx q96[3], q96[6];
x q96[3];
cx q96[1], q96[3];
mcx q96[7], q96[5], q96[6], q96[8];
cx q96[1], q96[3];
x q96[3];
cx q96[3], q96[6];
x q96[3];
cx q96[1], q96[3];
cx q96[0], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[0], q96[2];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
ccx q96[5], q96[6], q96[7];
x q96[7];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
h q96[0];
x q96[0];
h q96[1];
x q96[1];
h q96[2];
x q96[2];
h q96[3];
x q96[3];
z q96[4];
mcx_129722869770848 q96[0], q96[1], q96[2], q96[3], q96[4];
z q96[4];
x q96[0];
h q96[0];
x q96[1];
h q96[1];
x q96[2];
h q96[2];
x q96[3];
h q96[3];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
x q96[7];
ccx q96[5], q96[6], q96[7];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[0], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[0], q96[2];
cx q96[1], q96[3];
x q96[3];
cx q96[3], q96[6];
x q96[3];
cx q96[1], q96[3];
mcx q96[7], q96[5], q96[6], q96[8];
cx q96[1], q96[3];
x q96[3];
cx q96[3], q96[6];
x q96[3];
cx q96[1], q96[3];
cx q96[0], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[0], q96[2];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
ccx q96[5], q96[6], q96[7];
x q96[7];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
h q96[0];
x q96[0];
h q96[1];
x q96[1];
h q96[2];
x q96[2];
h q96[3];
x q96[3];
z q96[4];
mcx_129722950344224 q96[0], q96[1], q96[2], q96[3], q96[4];
z q96[4];
x q96[0];
h q96[0];
x q96[1];
h q96[1];
x q96[2];
h q96[2];
x q96[3];
h q96[3];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
x q96[7];
ccx q96[5], q96[6], q96[7];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[0], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[0], q96[2];
cx q96[1], q96[3];
x q96[3];
cx q96[3], q96[6];
x q96[3];
cx q96[1], q96[3];
mcx q96[7], q96[5], q96[6], q96[8];
cx q96[1], q96[3];
x q96[3];
cx q96[3], q96[6];
x q96[3];
cx q96[1], q96[3];
cx q96[0], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[0], q96[2];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
ccx q96[5], q96[6], q96[7];
x q96[7];
cx q96[3], q96[4];
x q96[4];
cx q96[4], q96[6];
x q96[4];
cx q96[3], q96[4];
cx q96[1], q96[2];
x q96[2];
cx q96[2], q96[5];
x q96[2];
cx q96[1], q96[2];
h q96[0];
x q96[0];
h q96[1];
x q96[1];
h q96[2];
x q96[2];
h q96[3];
x q96[3];
z q96[4];
mcx_129723051310816 q96[0], q96[1], q96[2], q96[3], q96[4];
z q96[4];
x q96[0];
h q96[0];
x q96[1];
h q96[1];
x q96[2];
h q96[2];
x q96[3];
h q96[3];
_creg[0] = measure q96[0];
_creg[1] = measure q96[1];
_creg[2] = measure q96[2];
_creg[3] = measure q96[3];
_creg[4] = measure q96[4];