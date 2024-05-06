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
gate cu1_129723063429584(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129723063429776(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722863619376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722863616688(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722863616400(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722863616640(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722863617888(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722863619232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722863620048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722863619376(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722863616688(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722863616400(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722863616640(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722863617888(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722863619232(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722863620048(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx_129723045654640 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129723063429584(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129723063429776(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722957147584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
}
gate cu1_129722957145328(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722957146864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
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
gate cu1_129722957146912(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957145184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957145760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957144464(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722957145616(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722648107760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722648105024(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722648105552(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722648106608(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate cu1_129722648105792(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/32) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/32) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/32) _gate_q_1;
}
gate mcu1_129722957144176(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  cu1_129722957147584(pi/16) _gate_q_4, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129722957145328(-pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_4, _gate_q_3;
  cu1_129722957147584(pi/16) _gate_q_3, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722957144656(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129722957147584(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_3, _gate_q_2;
  cu1_129722957146864(-pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_4, _gate_q_2;
  cu1_129722957147584(pi/16) _gate_q_2, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722957144320(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722957147344(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_2, _gate_q_1;
  cu1_129722957146912(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_3, _gate_q_1;
  cu1_129722957145184(-pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_4, _gate_q_1;
  cu1_129722957147584(pi/16) _gate_q_1, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722957145760(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722957144464(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722957145616(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722648107760(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_1, _gate_q_0;
  cu1_129722648105024(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722648105552(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_2, _gate_q_0;
  cu1_129722648106608(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_3, _gate_q_0;
  cu1_129722648105792(-pi/16) _gate_q_0, _gate_q_5;
  cx _gate_q_4, _gate_q_0;
  cu1_129722957147584(pi/16) _gate_q_0, _gate_q_5;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5 {
  h _gate_q_5;
  mcu1_129722957144176(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5;
  h _gate_q_5;
}
gate cu1_129722863771168(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722863770064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129723064861792 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722856892832 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722863771168(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722863770064(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129723064861792 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
bit[6] _creg;
qubit[11] q109;
h q109[0];
h q109[1];
h q109[2];
h q109[3];
h q109[4];
h q109[5];
x q109[10];
h q109[10];
x q109[6];
x q109[7];
x q109[8];
ccx q109[0], q109[1], q109[6];
ccx q109[2], q109[3], q109[7];
ccx q109[4], q109[5], q109[8];
mcx q109[6], q109[7], q109[8], q109[9];
ccx q109[4], q109[5], q109[8];
ccx q109[2], q109[3], q109[7];
ccx q109[0], q109[1], q109[6];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[6];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[7];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
x q109[8];
ccx q109[6], q109[7], q109[8];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[7];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
cx q109[2], q109[4];
x q109[4];
cx q109[3], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[3], q109[5];
x q109[4];
cx q109[2], q109[4];
mcx_129723045654640 q109[6], q109[7], q109[8], q109[9], q109[10];
cx q109[2], q109[4];
x q109[4];
cx q109[3], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[3], q109[5];
x q109[4];
cx q109[2], q109[4];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[7];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
x q109[8];
ccx q109[6], q109[7], q109[8];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[7];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[6];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
ccx q109[0], q109[1], q109[6];
ccx q109[2], q109[3], q109[7];
ccx q109[4], q109[5], q109[8];
mcx q109[6], q109[7], q109[8], q109[9];
ccx q109[4], q109[5], q109[8];
ccx q109[2], q109[3], q109[7];
ccx q109[0], q109[1], q109[6];
h q109[0];
x q109[0];
h q109[1];
x q109[1];
h q109[2];
x q109[2];
h q109[3];
x q109[3];
h q109[4];
x q109[4];
z q109[5];
mcx_gray q109[0], q109[1], q109[2], q109[3], q109[4], q109[5];
z q109[5];
x q109[0];
h q109[0];
x q109[1];
h q109[1];
x q109[2];
h q109[2];
x q109[3];
h q109[3];
x q109[4];
h q109[4];
ccx q109[0], q109[1], q109[6];
ccx q109[2], q109[3], q109[7];
ccx q109[4], q109[5], q109[8];
mcx q109[6], q109[7], q109[8], q109[9];
ccx q109[4], q109[5], q109[8];
ccx q109[2], q109[3], q109[7];
ccx q109[0], q109[1], q109[6];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[6];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[7];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
x q109[8];
ccx q109[6], q109[7], q109[8];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[7];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
cx q109[2], q109[4];
x q109[4];
cx q109[3], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[3], q109[5];
x q109[4];
cx q109[2], q109[4];
mcx_129722856892832 q109[6], q109[7], q109[8], q109[9], q109[10];
cx q109[2], q109[4];
x q109[4];
cx q109[3], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[3], q109[5];
x q109[4];
cx q109[2], q109[4];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[7];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[6];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
x q109[8];
ccx q109[6], q109[7], q109[8];
cx q109[0], q109[4];
x q109[4];
cx q109[1], q109[5];
x q109[5];
ccx q109[4], q109[5], q109[7];
x q109[5];
cx q109[1], q109[5];
x q109[4];
cx q109[0], q109[4];
cx q109[0], q109[2];
x q109[2];
cx q109[1], q109[3];
x q109[3];
ccx q109[2], q109[3], q109[6];
x q109[3];
cx q109[1], q109[3];
x q109[2];
cx q109[0], q109[2];
ccx q109[0], q109[1], q109[6];
ccx q109[2], q109[3], q109[7];
ccx q109[4], q109[5], q109[8];
mcx q109[6], q109[7], q109[8], q109[9];
ccx q109[4], q109[5], q109[8];
ccx q109[2], q109[3], q109[7];
ccx q109[0], q109[1], q109[6];
h q109[0];
x q109[0];
h q109[1];
x q109[1];
h q109[2];
x q109[2];
h q109[3];
x q109[3];
h q109[4];
x q109[4];
z q109[5];
mcx_gray q109[0], q109[1], q109[2], q109[3], q109[4], q109[5];
z q109[5];
x q109[0];
h q109[0];
x q109[1];
h q109[1];
x q109[2];
h q109[2];
x q109[3];
h q109[3];
x q109[4];
h q109[4];
_creg[0] = measure q109[0];
_creg[1] = measure q109[1];
_creg[2] = measure q109[2];
_creg[3] = measure q109[3];
_creg[4] = measure q109[4];
_creg[5] = measure q109[5];
