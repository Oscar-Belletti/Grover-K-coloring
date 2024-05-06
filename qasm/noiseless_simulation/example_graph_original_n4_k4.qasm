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
gate cu1_129722950311552(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722648106368(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722976460656(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722976459696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722976459456(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722976457344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722976460512(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129723064496432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129723064498256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722976460656(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722976459696(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722976459456(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722976457344(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722976460512(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129723064496432(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129723064498256(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx_129723064250864 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722950311552(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722648106368(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129723045674384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
}
gate cu1_129723045673856(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045672368(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045674768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045672416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045673424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045671792(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045674288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045673904(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723045673664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859549312(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859549264(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859551760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859550176(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859552672(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859550944(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859550848(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859552384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859552576(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859550704(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722859552144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958396480(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958394224(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958393696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958396768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958393840(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958397104(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958395232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958396912(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958396384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958393888(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958394608(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958395664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958396672(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064328096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064328144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064326416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064326896(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064327568(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064327760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064326320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064327184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064327328(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064324448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064325024(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064327280(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064324976(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723064325504(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976546672(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976546384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976542976(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976546288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976544800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976544224(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976544032(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976546528(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976544608(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976545136(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976543456(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976544512(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722976545568(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723044948096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723044949968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129723044948240(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate mcu1_129723045749520(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  cu1_129723045674384(pi/64) _gate_q_6, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129723045673856(-pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129723045674384(pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129723045672368(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129723045674384(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129723045674768(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129723045674384(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129723045672416(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129723045674384(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129723045673424(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129723045674384(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129723045671792(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129723045674384(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129723045674288(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129723045674384(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129723045673904(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129723045673664(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722859549312(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722859549264(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129722859551760(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722859550176(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722859552672(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722859550944(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129723045674384(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722859550848(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722859552384(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722859552576(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722859550704(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722859552144(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958396480(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722958394224(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958393696(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722958396768(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958393840(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722958397104(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958395232(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722958396912(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958396384(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722958393888(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958394608(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129723045674384(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722958395664(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722958396672(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064328096(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064328144(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129723064326416(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064326896(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064327568(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064327760(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129723064326320(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064327184(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064327328(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064324448(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129723064325024(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064327280(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723064324976(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723064325504(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722976546672(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722976546384(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722976542976(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722976546288(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722976544800(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722976544224(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722976544032(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722976546528(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129722976544608(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722976545136(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722976543456(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722976544512(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722976545568(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723044948096(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129723044949968(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129723044948240(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129723045674384(pi/64) _gate_q_0, _gate_q_7;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  h _gate_q_7;
  mcu1_129723045749520(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7;
  h _gate_q_7;
}
gate cu1_129722957147488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129723045751344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129723045750048 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722856891104 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722957147488(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129723045751344(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129723045750048 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
bit[8] _creg;
qubit[13] q112;
h q112[0];
h q112[1];
h q112[2];
h q112[3];
h q112[4];
h q112[5];
h q112[6];
h q112[7];
x q112[12];
h q112[12];
x q112[8];
x q112[9];
x q112[10];
x q112[11];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[8];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[10];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
x q112[11];
mcx q112[8], q112[9], q112[10], q112[11];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[10];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[8];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[9];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
x q112[10];
ccx q112[8], q112[9], q112[10];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[4], q112[6];
x q112[6];
cx q112[5], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[5], q112[7];
x q112[6];
cx q112[4], q112[6];
mcx_129723064250864 q112[8], q112[9], q112[10], q112[11], q112[12];
cx q112[4], q112[6];
x q112[6];
cx q112[5], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[5], q112[7];
x q112[6];
cx q112[4], q112[6];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
x q112[10];
ccx q112[8], q112[9], q112[10];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[9];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[8];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[10];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
x q112[11];
mcx q112[8], q112[9], q112[10], q112[11];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[10];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[8];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
h q112[0];
x q112[0];
h q112[1];
x q112[1];
h q112[2];
x q112[2];
h q112[3];
x q112[3];
h q112[4];
x q112[4];
h q112[5];
x q112[5];
h q112[6];
x q112[6];
z q112[7];
mcx_gray q112[0], q112[1], q112[2], q112[3], q112[4], q112[5], q112[6], q112[7];
z q112[7];
x q112[0];
h q112[0];
x q112[1];
h q112[1];
x q112[2];
h q112[2];
x q112[3];
h q112[3];
x q112[4];
h q112[4];
x q112[5];
h q112[5];
x q112[6];
h q112[6];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[8];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[10];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
x q112[11];
mcx q112[8], q112[9], q112[10], q112[11];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[10];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[8];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[9];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
x q112[10];
ccx q112[8], q112[9], q112[10];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[4], q112[6];
x q112[6];
cx q112[5], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[5], q112[7];
x q112[6];
cx q112[4], q112[6];
mcx_129722856891104 q112[8], q112[9], q112[10], q112[11], q112[12];
cx q112[4], q112[6];
x q112[6];
cx q112[5], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[5], q112[7];
x q112[6];
cx q112[4], q112[6];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
x q112[10];
ccx q112[8], q112[9], q112[10];
cx q112[2], q112[6];
x q112[6];
cx q112[3], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[9];
x q112[7];
cx q112[3], q112[7];
x q112[6];
cx q112[2], q112[6];
cx q112[2], q112[4];
x q112[4];
cx q112[3], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[8];
x q112[5];
cx q112[3], q112[5];
x q112[4];
cx q112[2], q112[4];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[10];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[8];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
x q112[11];
mcx q112[8], q112[9], q112[10], q112[11];
cx q112[0], q112[6];
x q112[6];
cx q112[1], q112[7];
x q112[7];
ccx q112[6], q112[7], q112[10];
x q112[7];
cx q112[1], q112[7];
x q112[6];
cx q112[0], q112[6];
cx q112[0], q112[4];
x q112[4];
cx q112[1], q112[5];
x q112[5];
ccx q112[4], q112[5], q112[9];
x q112[5];
cx q112[1], q112[5];
x q112[4];
cx q112[0], q112[4];
cx q112[0], q112[2];
x q112[2];
cx q112[1], q112[3];
x q112[3];
ccx q112[2], q112[3], q112[8];
x q112[3];
cx q112[1], q112[3];
x q112[2];
cx q112[0], q112[2];
h q112[0];
x q112[0];
h q112[1];
x q112[1];
h q112[2];
x q112[2];
h q112[3];
x q112[3];
h q112[4];
x q112[4];
h q112[5];
x q112[5];
h q112[6];
x q112[6];
z q112[7];
mcx_gray q112[0], q112[1], q112[2], q112[3], q112[4], q112[5], q112[6], q112[7];
z q112[7];
x q112[0];
h q112[0];
x q112[1];
h q112[1];
x q112[2];
h q112[2];
x q112[3];
h q112[3];
x q112[4];
h q112[4];
x q112[5];
h q112[5];
x q112[6];
h q112[6];
_creg[0] = measure q112[0];
_creg[1] = measure q112[1];
_creg[2] = measure q112[2];
_creg[3] = measure q112[3];
_creg[4] = measure q112[4];
_creg[5] = measure q112[5];
_creg[6] = measure q112[6];
_creg[7] = measure q112[7];
