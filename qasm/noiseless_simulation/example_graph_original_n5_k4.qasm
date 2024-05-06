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
gate cu1_129722957147536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
}
gate cu1_129722957144320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957147344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957145040(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957148112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957144656(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649278160(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649279216(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277728(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649280176(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649279408(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649278976(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722649277056(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722863618896(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate mcu1_129722957146048(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  cu1_129722957147536(pi/16) _gate_q_4, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129722957144320(-pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129722957147536(pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722957147344(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129722957147536(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722957145040(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129722957147536(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722957148112(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147536(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722957144656(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147536(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722649277392(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147536(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722649278160(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147536(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722649279216(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722649277728(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722649280176(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722649277488(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722649279408(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722649278976(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722649277056(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722863618896(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147536(pi/16) _gate_q_0, _gate_q_5;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  h _gate_q_5;
  mcu1_129722957146048(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5;
  h _gate_q_5;
}
bit[10] _creg;
qubit[16] q115;
h q115[0];
h q115[1];
h q115[2];
h q115[3];
h q115[4];
h q115[5];
h q115[6];
h q115[7];
h q115[8];
h q115[9];
x q115[15];
h q115[15];
x q115[10];
x q115[11];
x q115[12];
x q115[13];
x q115[14];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[10];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[12];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
x q115[14];
mcx q115[10], q115[11], q115[12], q115[14];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[12];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[10];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[12];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
x q115[13];
mcx q115[10], q115[11], q115[12], q115[13];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[10];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[12];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[4], q115[6];
x q115[6];
cx q115[5], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[5], q115[7];
x q115[6];
cx q115[4], q115[6];
mcx_gray q115[10], q115[11], q115[12], q115[13], q115[14], q115[15];
cx q115[4], q115[6];
x q115[6];
cx q115[5], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[5], q115[7];
x q115[6];
cx q115[4], q115[6];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[12];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[10];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
x q115[13];
mcx q115[10], q115[11], q115[12], q115[13];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[12];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[10];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[12];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
x q115[14];
mcx q115[10], q115[11], q115[12], q115[14];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[12];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[10];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
h q115[0];
x q115[0];
h q115[1];
x q115[1];
h q115[2];
x q115[2];
h q115[3];
x q115[3];
h q115[4];
x q115[4];
h q115[5];
x q115[5];
h q115[6];
x q115[6];
h q115[7];
x q115[7];
h q115[8];
x q115[8];
z q115[9];
mcx_gray q115[0], q115[1], q115[2], q115[3], q115[4], q115[5], q115[6], q115[7], q115[8], q115[9];
z q115[9];
x q115[0];
h q115[0];
x q115[1];
h q115[1];
x q115[2];
h q115[2];
x q115[3];
h q115[3];
x q115[4];
h q115[4];
x q115[5];
h q115[5];
x q115[6];
h q115[6];
x q115[7];
h q115[7];
x q115[8];
h q115[8];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[10];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[12];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
x q115[14];
mcx q115[10], q115[11], q115[12], q115[14];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[12];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[10];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[12];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
x q115[13];
mcx q115[10], q115[11], q115[12], q115[13];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[10];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[12];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[4], q115[6];
x q115[6];
cx q115[5], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[5], q115[7];
x q115[6];
cx q115[4], q115[6];
mcx_gray q115[10], q115[11], q115[12], q115[13], q115[14], q115[15];
cx q115[4], q115[6];
x q115[6];
cx q115[5], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[5], q115[7];
x q115[6];
cx q115[4], q115[6];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[12];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[10];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
x q115[13];
mcx q115[10], q115[11], q115[12], q115[13];
cx q115[2], q115[8];
x q115[8];
cx q115[3], q115[9];
x q115[9];
ccx q115[8], q115[9], q115[12];
x q115[9];
cx q115[3], q115[9];
x q115[8];
cx q115[2], q115[8];
cx q115[2], q115[6];
x q115[6];
cx q115[3], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[11];
x q115[7];
cx q115[3], q115[7];
x q115[6];
cx q115[2], q115[6];
cx q115[2], q115[4];
x q115[4];
cx q115[3], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[10];
x q115[5];
cx q115[3], q115[5];
x q115[4];
cx q115[2], q115[4];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[12];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[10];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
x q115[14];
mcx q115[10], q115[11], q115[12], q115[14];
cx q115[0], q115[6];
x q115[6];
cx q115[1], q115[7];
x q115[7];
ccx q115[6], q115[7], q115[12];
x q115[7];
cx q115[1], q115[7];
x q115[6];
cx q115[0], q115[6];
cx q115[0], q115[4];
x q115[4];
cx q115[1], q115[5];
x q115[5];
ccx q115[4], q115[5], q115[11];
x q115[5];
cx q115[1], q115[5];
x q115[4];
cx q115[0], q115[4];
cx q115[0], q115[2];
x q115[2];
cx q115[1], q115[3];
x q115[3];
ccx q115[2], q115[3], q115[10];
x q115[3];
cx q115[1], q115[3];
x q115[2];
cx q115[0], q115[2];
h q115[0];
x q115[0];
h q115[1];
x q115[1];
h q115[2];
x q115[2];
h q115[3];
x q115[3];
h q115[4];
x q115[4];
h q115[5];
x q115[5];
h q115[6];
x q115[6];
h q115[7];
x q115[7];
h q115[8];
x q115[8];
z q115[9];
mcx_gray q115[0], q115[1], q115[2], q115[3], q115[4], q115[5], q115[6], q115[7], q115[8], q115[9];
z q115[9];
x q115[0];
h q115[0];
x q115[1];
h q115[1];
x q115[2];
h q115[2];
x q115[3];
h q115[3];
x q115[4];
h q115[4];
x q115[5];
h q115[5];
x q115[6];
h q115[6];
x q115[7];
h q115[7];
x q115[8];
h q115[8];
_creg[0] = measure q115[0];
_creg[1] = measure q115[1];
_creg[2] = measure q115[2];
_creg[3] = measure q115[3];
_creg[4] = measure q115[4];
_creg[5] = measure q115[5];
_creg[6] = measure q115[6];
_creg[7] = measure q115[7];
_creg[8] = measure q115[8];
_creg[9] = measure q115[9];
