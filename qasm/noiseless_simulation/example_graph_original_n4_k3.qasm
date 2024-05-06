OPENQASM 3;
include "stdgates.inc";
gate cu1_129723045665808(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129723045666240(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722649279360(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722649279216(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722649278784(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722649276912(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722649277584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722943461360(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722943460448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722649279360(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722649279216(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722649278784(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722649276912(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722649277584(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722943461360(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722943460448(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129723045665808(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129723045666240(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129723051325184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
}
gate cu1_129723051327344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964236128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964236272(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964237328(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964234832(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964237472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964237664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964234880(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964237040(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964236320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964238288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964237424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964236080(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964235936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722964234688(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate mcu1_129723051324224(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  cu1_129723051325184(pi/16) _gate_q_4, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129723051327344(-pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129723051325184(pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722964236128(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129723051325184(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722964236272(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129723051325184(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722964237328(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129723051325184(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722964234832(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129723051325184(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722964237472(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129723051325184(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722964237664(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129723051325184(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722964234880(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722964237040(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722964236320(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722964238288(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722964237424(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722964236080(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722964235936(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722964234688(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129723051325184(pi/16) _gate_q_0, _gate_q_5;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  h _gate_q_5;
  mcu1_129723051324224(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5;
  h _gate_q_5;
}
bit[8] _creg;
qubit[14] q111;
h q111[0];
h q111[1];
h q111[2];
h q111[3];
h q111[4];
h q111[5];
h q111[6];
h q111[7];
x q111[13];
h q111[13];
x q111[8];
x q111[9];
x q111[10];
x q111[11];
ccx q111[0], q111[1], q111[8];
ccx q111[2], q111[3], q111[9];
ccx q111[4], q111[5], q111[10];
ccx q111[6], q111[7], q111[11];
mcx q111[8], q111[9], q111[10], q111[11], q111[12];
ccx q111[6], q111[7], q111[11];
ccx q111[4], q111[5], q111[10];
ccx q111[2], q111[3], q111[9];
ccx q111[0], q111[1], q111[8];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
x q111[11];
ccx q111[8], q111[9], q111[11];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[8];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[9];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[2], q111[4];
x q111[4];
cx q111[3], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[3], q111[5];
x q111[4];
cx q111[2], q111[4];
cx q111[4], q111[6];
x q111[6];
cx q111[5], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[5], q111[7];
x q111[6];
cx q111[4], q111[6];
mcx_gray q111[8], q111[9], q111[10], q111[11], q111[12], q111[13];
cx q111[4], q111[6];
x q111[6];
cx q111[5], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[5], q111[7];
x q111[6];
cx q111[4], q111[6];
cx q111[2], q111[4];
x q111[4];
cx q111[3], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[3], q111[5];
x q111[4];
cx q111[2], q111[4];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[9];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[8];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
x q111[11];
ccx q111[8], q111[9], q111[11];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
ccx q111[0], q111[1], q111[8];
ccx q111[2], q111[3], q111[9];
ccx q111[4], q111[5], q111[10];
ccx q111[6], q111[7], q111[11];
mcx q111[8], q111[9], q111[10], q111[11], q111[12];
ccx q111[6], q111[7], q111[11];
ccx q111[4], q111[5], q111[10];
ccx q111[2], q111[3], q111[9];
ccx q111[0], q111[1], q111[8];
h q111[0];
x q111[0];
h q111[1];
x q111[1];
h q111[2];
x q111[2];
h q111[3];
x q111[3];
h q111[4];
x q111[4];
h q111[5];
x q111[5];
h q111[6];
x q111[6];
z q111[7];
mcx_gray q111[0], q111[1], q111[2], q111[3], q111[4], q111[5], q111[6], q111[7];
z q111[7];
x q111[0];
h q111[0];
x q111[1];
h q111[1];
x q111[2];
h q111[2];
x q111[3];
h q111[3];
x q111[4];
h q111[4];
x q111[5];
h q111[5];
x q111[6];
h q111[6];
ccx q111[0], q111[1], q111[8];
ccx q111[2], q111[3], q111[9];
ccx q111[4], q111[5], q111[10];
ccx q111[6], q111[7], q111[11];
mcx q111[8], q111[9], q111[10], q111[11], q111[12];
ccx q111[6], q111[7], q111[11];
ccx q111[4], q111[5], q111[10];
ccx q111[2], q111[3], q111[9];
ccx q111[0], q111[1], q111[8];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
x q111[11];
ccx q111[8], q111[9], q111[11];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[8];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[9];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[2], q111[4];
x q111[4];
cx q111[3], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[3], q111[5];
x q111[4];
cx q111[2], q111[4];
cx q111[4], q111[6];
x q111[6];
cx q111[5], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[5], q111[7];
x q111[6];
cx q111[4], q111[6];
mcx_gray q111[8], q111[9], q111[10], q111[11], q111[12], q111[13];
cx q111[4], q111[6];
x q111[6];
cx q111[5], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[5], q111[7];
x q111[6];
cx q111[4], q111[6];
cx q111[2], q111[4];
x q111[4];
cx q111[3], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[3], q111[5];
x q111[4];
cx q111[2], q111[4];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[9];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[8];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
x q111[11];
ccx q111[8], q111[9], q111[11];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
ccx q111[0], q111[1], q111[8];
ccx q111[2], q111[3], q111[9];
ccx q111[4], q111[5], q111[10];
ccx q111[6], q111[7], q111[11];
mcx q111[8], q111[9], q111[10], q111[11], q111[12];
ccx q111[6], q111[7], q111[11];
ccx q111[4], q111[5], q111[10];
ccx q111[2], q111[3], q111[9];
ccx q111[0], q111[1], q111[8];
h q111[0];
x q111[0];
h q111[1];
x q111[1];
h q111[2];
x q111[2];
h q111[3];
x q111[3];
h q111[4];
x q111[4];
h q111[5];
x q111[5];
h q111[6];
x q111[6];
z q111[7];
mcx_gray q111[0], q111[1], q111[2], q111[3], q111[4], q111[5], q111[6], q111[7];
z q111[7];
x q111[0];
h q111[0];
x q111[1];
h q111[1];
x q111[2];
h q111[2];
x q111[3];
h q111[3];
x q111[4];
h q111[4];
x q111[5];
h q111[5];
x q111[6];
h q111[6];
ccx q111[0], q111[1], q111[8];
ccx q111[2], q111[3], q111[9];
ccx q111[4], q111[5], q111[10];
ccx q111[6], q111[7], q111[11];
mcx q111[8], q111[9], q111[10], q111[11], q111[12];
ccx q111[6], q111[7], q111[11];
ccx q111[4], q111[5], q111[10];
ccx q111[2], q111[3], q111[9];
ccx q111[0], q111[1], q111[8];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
x q111[11];
ccx q111[8], q111[9], q111[11];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[8];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[9];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[2], q111[4];
x q111[4];
cx q111[3], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[3], q111[5];
x q111[4];
cx q111[2], q111[4];
cx q111[4], q111[6];
x q111[6];
cx q111[5], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[5], q111[7];
x q111[6];
cx q111[4], q111[6];
mcx_gray q111[8], q111[9], q111[10], q111[11], q111[12], q111[13];
cx q111[4], q111[6];
x q111[6];
cx q111[5], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[5], q111[7];
x q111[6];
cx q111[4], q111[6];
cx q111[2], q111[4];
x q111[4];
cx q111[3], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[3], q111[5];
x q111[4];
cx q111[2], q111[4];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[9];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[8];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
x q111[11];
ccx q111[8], q111[9], q111[11];
cx q111[0], q111[6];
x q111[6];
cx q111[1], q111[7];
x q111[7];
ccx q111[6], q111[7], q111[9];
x q111[7];
cx q111[1], q111[7];
x q111[6];
cx q111[0], q111[6];
cx q111[0], q111[4];
x q111[4];
cx q111[1], q111[5];
x q111[5];
ccx q111[4], q111[5], q111[8];
x q111[5];
cx q111[1], q111[5];
x q111[4];
cx q111[0], q111[4];
ccx q111[0], q111[1], q111[8];
ccx q111[2], q111[3], q111[9];
ccx q111[4], q111[5], q111[10];
ccx q111[6], q111[7], q111[11];
mcx q111[8], q111[9], q111[10], q111[11], q111[12];
ccx q111[6], q111[7], q111[11];
ccx q111[4], q111[5], q111[10];
ccx q111[2], q111[3], q111[9];
ccx q111[0], q111[1], q111[8];
h q111[0];
x q111[0];
h q111[1];
x q111[1];
h q111[2];
x q111[2];
h q111[3];
x q111[3];
h q111[4];
x q111[4];
h q111[5];
x q111[5];
h q111[6];
x q111[6];
z q111[7];
mcx_gray q111[0], q111[1], q111[2], q111[3], q111[4], q111[5], q111[6], q111[7];
z q111[7];
x q111[0];
h q111[0];
x q111[1];
h q111[1];
x q111[2];
h q111[2];
x q111[3];
h q111[3];
x q111[4];
h q111[4];
x q111[5];
h q111[5];
x q111[6];
h q111[6];
_creg[0] = measure q111[0];
_creg[1] = measure q111[1];
_creg[2] = measure q111[2];
_creg[3] = measure q111[3];
_creg[4] = measure q111[4];
_creg[5] = measure q111[5];
_creg[6] = measure q111[6];
_creg[7] = measure q111[7];
