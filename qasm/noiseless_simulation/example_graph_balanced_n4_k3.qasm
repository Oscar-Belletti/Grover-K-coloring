OPENQASM 3;
include "stdgates.inc";
gate cu1_129723064496528(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129723064497488(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722869769216(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722869770032(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722869770320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722869770560(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722869770944(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722869771952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722869772096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722869769216(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722869770032(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722869770320(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722869770560(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722869770944(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722869771952(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722869772096(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129723064496528(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129723064497488(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722649277584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
}
gate cu1_129722649278496(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649280128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649278784(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277920(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277296(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277248(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277008(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649276864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722855485696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722855485504(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722855486416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722855487136(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722855488240(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722855487952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate mcu1_129722649277536(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  cu1_129722649277584(pi/16) _gate_q_4, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129722649278496(-pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129722649277584(pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722649280128(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129722649277584(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722649278784(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129722649277584(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722649277920(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722649277584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722649277296(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722649277584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722649277776(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722649277584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722649277248(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722649277584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722649277008(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722649276864(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722855485696(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722855485504(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722855486416(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722855487136(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722855488240(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722855487952(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722649277584(pi/16) _gate_q_0, _gate_q_5;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  h _gate_q_5;
  mcu1_129722649277536(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5;
  h _gate_q_5;
}
bit[8] _creg;
qubit[14] q94;
h q94[0];
h q94[1];
h q94[2];
h q94[3];
h q94[4];
h q94[5];
h q94[6];
h q94[7];
x q94[13];
h q94[13];
x q94[8];
x q94[9];
x q94[10];
x q94[11];
x q94[12];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
ccx q94[6], q94[7], q94[11];
x q94[12];
mcx q94[8], q94[9], q94[10], q94[11], q94[12];
ccx q94[6], q94[7], q94[11];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[0], q94[6];
x q94[6];
cx q94[1], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[8];
x q94[7];
cx q94[1], q94[7];
x q94[6];
cx q94[0], q94[6];
ccx q94[2], q94[3], q94[9];
ccx q94[4], q94[5], q94[10];
ccx q94[0], q94[1], q94[11];
mcx_gray q94[12], q94[8], q94[9], q94[10], q94[11], q94[13];
ccx q94[0], q94[1], q94[11];
ccx q94[4], q94[5], q94[10];
ccx q94[2], q94[3], q94[9];
cx q94[0], q94[6];
x q94[6];
cx q94[1], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[8];
x q94[7];
cx q94[1], q94[7];
x q94[6];
cx q94[0], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
ccx q94[6], q94[7], q94[11];
mcx q94[8], q94[9], q94[10], q94[11], q94[12];
x q94[12];
ccx q94[6], q94[7], q94[11];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
h q94[0];
x q94[0];
h q94[1];
x q94[1];
h q94[2];
x q94[2];
h q94[3];
x q94[3];
h q94[4];
x q94[4];
h q94[5];
x q94[5];
h q94[6];
x q94[6];
z q94[7];
mcx_gray q94[0], q94[1], q94[2], q94[3], q94[4], q94[5], q94[6], q94[7];
z q94[7];
x q94[0];
h q94[0];
x q94[1];
h q94[1];
x q94[2];
h q94[2];
x q94[3];
h q94[3];
x q94[4];
h q94[4];
x q94[5];
h q94[5];
x q94[6];
h q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
ccx q94[6], q94[7], q94[11];
x q94[12];
mcx q94[8], q94[9], q94[10], q94[11], q94[12];
ccx q94[6], q94[7], q94[11];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[0], q94[6];
x q94[6];
cx q94[1], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[8];
x q94[7];
cx q94[1], q94[7];
x q94[6];
cx q94[0], q94[6];
ccx q94[2], q94[3], q94[9];
ccx q94[4], q94[5], q94[10];
ccx q94[0], q94[1], q94[11];
mcx_gray q94[12], q94[8], q94[9], q94[10], q94[11], q94[13];
ccx q94[0], q94[1], q94[11];
ccx q94[4], q94[5], q94[10];
ccx q94[2], q94[3], q94[9];
cx q94[0], q94[6];
x q94[6];
cx q94[1], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[8];
x q94[7];
cx q94[1], q94[7];
x q94[6];
cx q94[0], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
ccx q94[6], q94[7], q94[11];
mcx q94[8], q94[9], q94[10], q94[11], q94[12];
x q94[12];
ccx q94[6], q94[7], q94[11];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
h q94[0];
x q94[0];
h q94[1];
x q94[1];
h q94[2];
x q94[2];
h q94[3];
x q94[3];
h q94[4];
x q94[4];
h q94[5];
x q94[5];
h q94[6];
x q94[6];
z q94[7];
mcx_gray q94[0], q94[1], q94[2], q94[3], q94[4], q94[5], q94[6], q94[7];
z q94[7];
x q94[0];
h q94[0];
x q94[1];
h q94[1];
x q94[2];
h q94[2];
x q94[3];
h q94[3];
x q94[4];
h q94[4];
x q94[5];
h q94[5];
x q94[6];
h q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
ccx q94[6], q94[7], q94[11];
x q94[12];
mcx q94[8], q94[9], q94[10], q94[11], q94[12];
ccx q94[6], q94[7], q94[11];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[0], q94[6];
x q94[6];
cx q94[1], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[8];
x q94[7];
cx q94[1], q94[7];
x q94[6];
cx q94[0], q94[6];
ccx q94[2], q94[3], q94[9];
ccx q94[4], q94[5], q94[10];
ccx q94[0], q94[1], q94[11];
mcx_gray q94[12], q94[8], q94[9], q94[10], q94[11], q94[13];
ccx q94[0], q94[1], q94[11];
ccx q94[4], q94[5], q94[10];
ccx q94[2], q94[3], q94[9];
cx q94[0], q94[6];
x q94[6];
cx q94[1], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[8];
x q94[7];
cx q94[1], q94[7];
x q94[6];
cx q94[0], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
ccx q94[6], q94[7], q94[11];
mcx q94[8], q94[9], q94[10], q94[11], q94[12];
x q94[12];
ccx q94[6], q94[7], q94[11];
cx q94[0], q94[4];
x q94[4];
cx q94[1], q94[5];
x q94[5];
ccx q94[4], q94[5], q94[10];
x q94[5];
cx q94[1], q94[5];
x q94[4];
cx q94[0], q94[4];
cx q94[4], q94[6];
x q94[6];
cx q94[5], q94[7];
x q94[7];
ccx q94[6], q94[7], q94[9];
x q94[7];
cx q94[5], q94[7];
x q94[6];
cx q94[4], q94[6];
cx q94[0], q94[2];
x q94[2];
cx q94[1], q94[3];
x q94[3];
ccx q94[2], q94[3], q94[8];
x q94[3];
cx q94[1], q94[3];
x q94[2];
cx q94[0], q94[2];
h q94[0];
x q94[0];
h q94[1];
x q94[1];
h q94[2];
x q94[2];
h q94[3];
x q94[3];
h q94[4];
x q94[4];
h q94[5];
x q94[5];
h q94[6];
x q94[6];
z q94[7];
mcx_gray q94[0], q94[1], q94[2], q94[3], q94[4], q94[5], q94[6], q94[7];
z q94[7];
x q94[0];
h q94[0];
x q94[1];
h q94[1];
x q94[2];
h q94[2];
x q94[3];
h q94[3];
x q94[4];
h q94[4];
x q94[5];
h q94[5];
x q94[6];
h q94[6];
_creg[0] = measure q94[0];
_creg[1] = measure q94[1];
_creg[2] = measure q94[2];
_creg[3] = measure q94[3];
_creg[4] = measure q94[4];
_creg[5] = measure q94[5];
_creg[6] = measure q94[6];
_creg[7] = measure q94[7];