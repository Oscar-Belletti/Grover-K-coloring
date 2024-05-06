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
gate cu1_129722716556688(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722716553760(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722976546048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722976544416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722976542832(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722976544560(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722976544320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722976544176(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722976545232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722976546048(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722976544416(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722976542832(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722976544560(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722976544320(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722976544176(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722976545232(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx_129723063877104 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722716556688(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722716553760(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722972563968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
}
gate cu1_129722972562720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722972561568(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722972562048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051309424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051310288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051310720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051310768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051310432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051310048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051307936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051308128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723051308944(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855488768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855489008(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855487760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855485936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855485648(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855486080(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855487184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855487424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855487040(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722855487616(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651496752(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651497952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651498192(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651497184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651498624(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651499104(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651499536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651499920(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651500016(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651499008(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722651498816(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948233344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948233008(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948233296(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948233728(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948233680(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948231424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948234928(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948234832(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948234592(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948232048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722948231328(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064864288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064864384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064863568(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064864192(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064861168(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064864624(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064863616(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064864336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064861744(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064863904(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064862848(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064862272(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064864672(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722817782112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722817782256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722817782544(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722817783504(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722817784128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722817783072(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate mcu1_129722972563296(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  cu1_129722972563968(pi/64) _gate_q_6, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129722972562720(-pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129722972563968(pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129722972561568(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129722972563968(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129722972562048(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129722972563968(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129723051309424(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722972563968(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129723051310288(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722972563968(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129723051310720(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722972563968(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129723051310768(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722972563968(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129723051310432(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129723051310048(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129723051307936(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129723051308128(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129723051308944(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722855488768(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722855489008(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722855487760(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722972563968(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722855485936(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722855485648(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722855486080(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722855487184(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722855487424(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722855487040(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722855487616(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722651496752(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722651497952(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722651498192(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722651497184(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722651498624(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722651499104(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722651499536(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722651499920(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722651500016(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722972563968(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722651499008(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722651498816(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722948233344(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722948233008(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722948233296(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722948233728(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722948233680(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722948231424(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129722948234928(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722948234832(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722948234592(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722948232048(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722948231328(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064864288(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064864384(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064863568(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129723064864192(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064861168(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064864624(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064863616(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129723064864336(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064861744(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064863904(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064862848(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129723064862272(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064864672(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722817782112(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722817782256(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722817782544(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722817783504(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722817784128(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722817783072(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722972563968(pi/64) _gate_q_0, _gate_q_7;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  h _gate_q_7;
  mcu1_129722972563296(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7;
  h _gate_q_7;
}
gate cu1_129722762607488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722762608016(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722762604944 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722964236560 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722762607488(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722762608016(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722762604944 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
bit[8] _creg;
qubit[13] q95;
h q95[0];
h q95[1];
h q95[2];
h q95[3];
h q95[4];
h q95[5];
h q95[6];
h q95[7];
x q95[12];
h q95[12];
x q95[8];
x q95[9];
x q95[10];
x q95[11];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
x q95[11];
mcx q95[8], q95[9], q95[10], q95[11];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
cx q95[2], q95[4];
x q95[4];
cx q95[3], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[8];
x q95[5];
cx q95[3], q95[5];
x q95[4];
cx q95[2], q95[4];
cx q95[0], q95[6];
x q95[6];
cx q95[1], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[1], q95[7];
x q95[6];
cx q95[0], q95[6];
cx q95[2], q95[6];
x q95[6];
cx q95[3], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[10];
x q95[7];
cx q95[3], q95[7];
x q95[6];
cx q95[2], q95[6];
mcx_129723063877104 q95[11], q95[8], q95[9], q95[10], q95[12];
cx q95[2], q95[6];
x q95[6];
cx q95[3], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[10];
x q95[7];
cx q95[3], q95[7];
x q95[6];
cx q95[2], q95[6];
cx q95[0], q95[6];
x q95[6];
cx q95[1], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[1], q95[7];
x q95[6];
cx q95[0], q95[6];
cx q95[2], q95[4];
x q95[4];
cx q95[3], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[8];
x q95[5];
cx q95[3], q95[5];
x q95[4];
cx q95[2], q95[4];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
mcx q95[8], q95[9], q95[10], q95[11];
x q95[11];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
h q95[0];
x q95[0];
h q95[1];
x q95[1];
h q95[2];
x q95[2];
h q95[3];
x q95[3];
h q95[4];
x q95[4];
h q95[5];
x q95[5];
h q95[6];
x q95[6];
z q95[7];
mcx_gray q95[0], q95[1], q95[2], q95[3], q95[4], q95[5], q95[6], q95[7];
z q95[7];
x q95[0];
h q95[0];
x q95[1];
h q95[1];
x q95[2];
h q95[2];
x q95[3];
h q95[3];
x q95[4];
h q95[4];
x q95[5];
h q95[5];
x q95[6];
h q95[6];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
x q95[11];
mcx q95[8], q95[9], q95[10], q95[11];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
cx q95[2], q95[4];
x q95[4];
cx q95[3], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[8];
x q95[5];
cx q95[3], q95[5];
x q95[4];
cx q95[2], q95[4];
cx q95[0], q95[6];
x q95[6];
cx q95[1], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[1], q95[7];
x q95[6];
cx q95[0], q95[6];
cx q95[2], q95[6];
x q95[6];
cx q95[3], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[10];
x q95[7];
cx q95[3], q95[7];
x q95[6];
cx q95[2], q95[6];
mcx_129722964236560 q95[11], q95[8], q95[9], q95[10], q95[12];
cx q95[2], q95[6];
x q95[6];
cx q95[3], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[10];
x q95[7];
cx q95[3], q95[7];
x q95[6];
cx q95[2], q95[6];
cx q95[0], q95[6];
x q95[6];
cx q95[1], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[1], q95[7];
x q95[6];
cx q95[0], q95[6];
cx q95[2], q95[4];
x q95[4];
cx q95[3], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[8];
x q95[5];
cx q95[3], q95[5];
x q95[4];
cx q95[2], q95[4];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
mcx q95[8], q95[9], q95[10], q95[11];
x q95[11];
cx q95[0], q95[4];
x q95[4];
cx q95[1], q95[5];
x q95[5];
ccx q95[4], q95[5], q95[10];
x q95[5];
cx q95[1], q95[5];
x q95[4];
cx q95[0], q95[4];
cx q95[4], q95[6];
x q95[6];
cx q95[5], q95[7];
x q95[7];
ccx q95[6], q95[7], q95[9];
x q95[7];
cx q95[5], q95[7];
x q95[6];
cx q95[4], q95[6];
cx q95[0], q95[2];
x q95[2];
cx q95[1], q95[3];
x q95[3];
ccx q95[2], q95[3], q95[8];
x q95[3];
cx q95[1], q95[3];
x q95[2];
cx q95[0], q95[2];
h q95[0];
x q95[0];
h q95[1];
x q95[1];
h q95[2];
x q95[2];
h q95[3];
x q95[3];
h q95[4];
x q95[4];
h q95[5];
x q95[5];
h q95[6];
x q95[6];
z q95[7];
mcx_gray q95[0], q95[1], q95[2], q95[3], q95[4], q95[5], q95[6], q95[7];
z q95[7];
x q95[0];
h q95[0];
x q95[1];
h q95[1];
x q95[2];
h q95[2];
x q95[3];
h q95[3];
x q95[4];
h q95[4];
x q95[5];
h q95[5];
x q95[6];
h q95[6];
_creg[0] = measure q95[0];
_creg[1] = measure q95[1];
_creg[2] = measure q95[2];
_creg[3] = measure q95[3];
_creg[4] = measure q95[4];
_creg[5] = measure q95[5];
_creg[6] = measure q95[6];
_creg[7] = measure q95[7];
