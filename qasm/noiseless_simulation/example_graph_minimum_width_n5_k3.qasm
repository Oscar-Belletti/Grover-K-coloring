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
gate cu1_129722949305152(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129722949306400(_gate_p_0) _gate_q_0, _gate_q_1 {
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
gate cu1_129716674598368(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722949321440(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722949324176(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722949321392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722949322736(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722949324320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722949322832(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129716674598368(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722949321440(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722949324176(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722949321392(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722949322736(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722949324320(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722949322832(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx_129722877494128 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722949305152(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722949306400(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722958125664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722958126816(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722943352896 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722950242800 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722958125664(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722958126816(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722943352896 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722943356880(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
}
gate cu1_129722949338256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949337440(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949337968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949338352(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949339120(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949340752(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949339984(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949340944(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949340800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722948054848(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722948053744(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722948054512(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722948054416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722948054272(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970247664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970247520(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970249536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970247760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970248624(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970250064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970250112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970248144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970248000(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970250784(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722970250640(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949366784(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949366592(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949366688(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949367360(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949368800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949369280(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949369040(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949369424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949369184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958149472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958150144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958149328(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958150480(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958148704(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958150048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958149088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958150336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958149040(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958150000(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958150672(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973888864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973889392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973889536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973889488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973890448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973891696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973891552(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973892368(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973891360(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973890544(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722973890112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949383504(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949382256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949383408(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949384608(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949385136(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949382448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949382736(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949385424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949385856(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949399504(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949399408(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949399216(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949400896(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949402144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949402192(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949401952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949400752(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958166432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958167344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958167440(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958165664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958168016(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958164608(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958166384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958165376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958165520(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958166624(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958165424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958167392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462019792(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462020272(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462020512(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958125232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462020416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462022336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462023440(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462023296(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462022432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462019696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722462021904(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949420704(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949420416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949419888(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949421088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949421952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949422864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949422144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949422528(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949422384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949427936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949427888(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949428464(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949429952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949428656(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949430864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949429664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949430336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949427504(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856345856(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856346096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856346960(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856348064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856347920(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856348736(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856348784(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856347968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856349552(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856348448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722856345952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958192112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958190384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958191584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958190912(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958188752(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958190432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958191488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958191104(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958192304(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958192448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958188704(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949464752(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949465664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949465424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949465184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949465856(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949467728(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949466240(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949468016(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949466912(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958209424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958209088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958212256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958210720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958210096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958211632(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958211584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958209808(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958210576(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958211776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722958211488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944783120(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944782496(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944783456(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944783312(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944783792(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944784800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944785088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944785520(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944785376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944785472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722944786384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949477520(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949477808(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949477088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949477760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949478960(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949478672(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949479200(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949479776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949479488(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949493856(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949494336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949495104(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949494096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949494528(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949496304(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949496448(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949496688(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949492800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957444480(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957443376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957446112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957446256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957446928(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957443520(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957444816(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957444192(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957445728(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957445776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957445248(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648543392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648544208(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648544640(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648543824(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648544064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648545984(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648546272(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648546128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648546176(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648546656(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722648547280(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949518480(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949518336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949517424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949518144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949519536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949520208(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949520592(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949520160(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949517760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949353632(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949353968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949354064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949355936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949357088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949355024(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949355744(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949355840(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949354256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949530528(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949530720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949530768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949531536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949532400(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949531104(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949532208(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722949533120(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957464432(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957467360(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957467120(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957463808(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957464192(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957464240(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957466064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957465392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957465776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957464864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957463616(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722957466400(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920485600(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920485024(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920486464(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920486320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920486224(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920487184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920488576(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920488288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920488528(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate cu1_129722920488480(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/512) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/512) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/512) _gate_q_1;
}
gate mcu1_129722943356688(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7, _gate_q_8, _gate_q_9 {
  cu1_129722943356880(pi/256) _gate_q_8, _gate_q_9;
  cx _gate_q_8, _gate_q_7;
  cu1_129722949338256(-pi/256) _gate_q_7, _gate_q_9;
  cx _gate_q_8, _gate_q_7;
  cu1_129722943356880(pi/256) _gate_q_7, _gate_q_9;
  cx _gate_q_7, _gate_q_6;
  cu1_129722949337440(-pi/256) _gate_q_6, _gate_q_9;
  cx _gate_q_8, _gate_q_6;
  cu1_129722943356880(pi/256) _gate_q_6, _gate_q_9;
  cx _gate_q_7, _gate_q_6;
  cu1_129722949337968(-pi/256) _gate_q_6, _gate_q_9;
  cx _gate_q_8, _gate_q_6;
  cu1_129722943356880(pi/256) _gate_q_6, _gate_q_9;
  cx _gate_q_6, _gate_q_5;
  cu1_129722949338352(-pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_8, _gate_q_5;
  cu1_129722943356880(pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_7, _gate_q_5;
  cu1_129722949339120(-pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_8, _gate_q_5;
  cu1_129722943356880(pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_6, _gate_q_5;
  cu1_129722949340752(-pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_8, _gate_q_5;
  cu1_129722943356880(pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_7, _gate_q_5;
  cu1_129722949339984(-pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_8, _gate_q_5;
  cu1_129722943356880(pi/256) _gate_q_5, _gate_q_9;
  cx _gate_q_5, _gate_q_4;
  cu1_129722949340944(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_7, _gate_q_4;
  cu1_129722949340800(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_6, _gate_q_4;
  cu1_129722948054848(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_7, _gate_q_4;
  cu1_129722948053744(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_5, _gate_q_4;
  cu1_129722948054512(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_7, _gate_q_4;
  cu1_129722948054416(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_6, _gate_q_4;
  cu1_129722948054272(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_7, _gate_q_4;
  cu1_129722970247664(-pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_8, _gate_q_4;
  cu1_129722943356880(pi/256) _gate_q_4, _gate_q_9;
  cx _gate_q_4, _gate_q_3;
  cu1_129722970247520(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722970249536(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_6, _gate_q_3;
  cu1_129722970247760(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722970248624(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_5, _gate_q_3;
  cu1_129722970250064(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722970250112(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_6, _gate_q_3;
  cu1_129722970248144(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722970248000(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_4, _gate_q_3;
  cu1_129722970250784(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722970250640(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949366784(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722949366592(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_5, _gate_q_3;
  cu1_129722949366688(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722949367360(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949368800(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_7, _gate_q_3;
  cu1_129722949369280(-pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_8, _gate_q_3;
  cu1_129722943356880(pi/256) _gate_q_3, _gate_q_9;
  cx _gate_q_3, _gate_q_2;
  cu1_129722949369040(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722949369424(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949369184(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722958149472(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_5, _gate_q_2;
  cu1_129722958150144(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722958149328(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722958150480(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722958148704(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_4, _gate_q_2;
  cu1_129722958150048(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722958149088(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722958150336(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722958149040(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_5, _gate_q_2;
  cu1_129722958150000(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722958150672(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722973888864(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722973889392(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_3, _gate_q_2;
  cu1_129722973889536(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722973889488(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722973890448(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722973891696(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_5, _gate_q_2;
  cu1_129722973891552(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722973892368(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722973891360(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722973890544(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_4, _gate_q_2;
  cu1_129722973890112(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722949383504(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949382256(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722949383408(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_5, _gate_q_2;
  cu1_129722949384608(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722949385136(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949382448(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_7, _gate_q_2;
  cu1_129722949382736(-pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_8, _gate_q_2;
  cu1_129722943356880(pi/256) _gate_q_2, _gate_q_9;
  cx _gate_q_2, _gate_q_1;
  cu1_129722949385424(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949385856(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949399504(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949399408(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949399216(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949400896(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949402144(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949402192(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_4, _gate_q_1;
  cu1_129722949401952(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949400752(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722958166432(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958167344(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958167440(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958165664(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722958168016(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958164608(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_3, _gate_q_1;
  cu1_129722958166384(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958165376(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722958165520(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958166624(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722958165424(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958167392(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722462019792(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722462020272(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_4, _gate_q_1;
  cu1_129722462020512(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958125232(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722462020416(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722462022336(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722462023440(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722462023296(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722462022432(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722462019696(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_2, _gate_q_1;
  cu1_129722462021904(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949420704(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949420416(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949419888(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949421088(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949421952(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949422864(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949422144(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_4, _gate_q_1;
  cu1_129722949422528(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949422384(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949427936(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949427888(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949428464(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949429952(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949428656(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949430864(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_3, _gate_q_1;
  cu1_129722949429664(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722949430336(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949427504(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722856345856(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722856346096(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722856346960(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722856348064(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722856347920(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_4, _gate_q_1;
  cu1_129722856348736(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722856348784(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722856347968(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722856349552(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_5, _gate_q_1;
  cu1_129722856348448(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722856345952(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_6, _gate_q_1;
  cu1_129722958192112(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_7, _gate_q_1;
  cu1_129722958190384(-pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_8, _gate_q_1;
  cu1_129722943356880(pi/256) _gate_q_1, _gate_q_9;
  cx _gate_q_1, _gate_q_0;
  cu1_129722958191584(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958190912(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722958188752(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958190432(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722958191488(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958191104(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722958192304(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958192448(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722958188704(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949464752(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949465664(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949465424(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949465184(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949465856(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949467728(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949466240(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_3, _gate_q_0;
  cu1_129722949468016(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949466912(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722958209424(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958209088(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722958212256(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958210720(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722958210096(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958211632(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722958211584(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958209808(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722958210576(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722958211776(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722958211488(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722944783120(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722944782496(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722944783456(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_2, _gate_q_0;
  cu1_129722944783312(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722944783792(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722944784800(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722944785088(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722944785520(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722944785376(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722944785472(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722944786384(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949477520(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949477808(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949477088(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949477760(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949478960(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949478672(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949479200(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949479776(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_3, _gate_q_0;
  cu1_129722949479488(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949493856(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949494336(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949495104(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949494096(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949494528(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949496304(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949496448(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949496688(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949492800(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722957444480(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957443376(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722957446112(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957446256(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722957446928(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957443520(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_1, _gate_q_0;
  cu1_129722957444816(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957444192(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722957445728(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957445776(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722957445248(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722648543392(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722648544208(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722648544640(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722648543824(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722648544064(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722648545984(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722648546272(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722648546128(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722648546176(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722648546656(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722648547280(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_3, _gate_q_0;
  cu1_129722949518480(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949518336(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949517424(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949518144(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949519536(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949520208(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949520592(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949520160(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949517760(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949353632(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949353968(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949354064(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949355936(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949357088(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949355024(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949355744(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_2, _gate_q_0;
  cu1_129722949355840(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949354256(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949530528(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949530720(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949530768(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949531536(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949532400(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949531104(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949532208(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722949533120(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722957464432(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957467360(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722957467120(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957463808(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722957464192(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957464240(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_3, _gate_q_0;
  cu1_129722957466064(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957465392(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722957465776(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957464864(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722957463616(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722957466400(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722920485600(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722920485024(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_4, _gate_q_0;
  cu1_129722920486464(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722920486320(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722920486224(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722920487184(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_5, _gate_q_0;
  cu1_129722920488576(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722920488288(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_6, _gate_q_0;
  cu1_129722920488528(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_7, _gate_q_0;
  cu1_129722920488480(-pi/256) _gate_q_0, _gate_q_9;
  cx _gate_q_8, _gate_q_0;
  cu1_129722943356880(pi/256) _gate_q_0, _gate_q_9;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7, _gate_q_8, _gate_q_9 {
  h _gate_q_9;
  mcu1_129722943356688(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7, _gate_q_8, _gate_q_9;
  h _gate_q_9;
}
gate cu1_129722957681232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722957682720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722885910832 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722891147584 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722957681232(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722957682720(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722885910832 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722885914240(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722885914096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722948877232 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722958444720 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722885914240(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722885914096(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722948877232 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722948892944(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722948893760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722957181376 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722957983312 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722948892944(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722948893760(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722957181376 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722887632736(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722887631728(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722887634608 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722949849040 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722887632736(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722887631728(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722887634608 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722948910288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722948919744(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722948920800 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722897440192 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722948910288(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722948919744(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722948920800 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722892224832(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722892225312(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722957198048 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722949714512 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722892224832(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722892225312(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722957198048 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722948940800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722948940368(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722948943200 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129722949650656 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722948940800(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722948940368(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722948943200 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722896854128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722896853840(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722948957136 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
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
gate mcx_129717054408928 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722896854128(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722896853840(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722948957136 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
bit[10] _creg;
qubit[16] q80;
h q80[0];
h q80[1];
h q80[2];
h q80[3];
h q80[4];
h q80[5];
h q80[6];
h q80[7];
h q80[8];
h q80[9];
x q80[15];
h q80[15];
x q80[10];
x q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722877494128 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[14], q80[12], q80[15];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722950242800 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
h q80[0];
x q80[0];
h q80[1];
x q80[1];
h q80[2];
x q80[2];
h q80[3];
x q80[3];
h q80[4];
x q80[4];
h q80[5];
x q80[5];
h q80[6];
x q80[6];
h q80[7];
x q80[7];
h q80[8];
x q80[8];
z q80[9];
mcx_gray q80[0], q80[1], q80[2], q80[3], q80[4], q80[5], q80[6], q80[7], q80[8], q80[9];
z q80[9];
x q80[0];
h q80[0];
x q80[1];
h q80[1];
x q80[2];
h q80[2];
x q80[3];
h q80[3];
x q80[4];
h q80[4];
x q80[5];
h q80[5];
x q80[6];
h q80[6];
x q80[7];
h q80[7];
x q80[8];
h q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722891147584 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[14], q80[12], q80[15];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722958444720 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
h q80[0];
x q80[0];
h q80[1];
x q80[1];
h q80[2];
x q80[2];
h q80[3];
x q80[3];
h q80[4];
x q80[4];
h q80[5];
x q80[5];
h q80[6];
x q80[6];
h q80[7];
x q80[7];
h q80[8];
x q80[8];
z q80[9];
mcx_gray q80[0], q80[1], q80[2], q80[3], q80[4], q80[5], q80[6], q80[7], q80[8], q80[9];
z q80[9];
x q80[0];
h q80[0];
x q80[1];
h q80[1];
x q80[2];
h q80[2];
x q80[3];
h q80[3];
x q80[4];
h q80[4];
x q80[5];
h q80[5];
x q80[6];
h q80[6];
x q80[7];
h q80[7];
x q80[8];
h q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722957983312 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[14], q80[12], q80[15];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722949849040 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
h q80[0];
x q80[0];
h q80[1];
x q80[1];
h q80[2];
x q80[2];
h q80[3];
x q80[3];
h q80[4];
x q80[4];
h q80[5];
x q80[5];
h q80[6];
x q80[6];
h q80[7];
x q80[7];
h q80[8];
x q80[8];
z q80[9];
mcx_gray q80[0], q80[1], q80[2], q80[3], q80[4], q80[5], q80[6], q80[7], q80[8], q80[9];
z q80[9];
x q80[0];
h q80[0];
x q80[1];
h q80[1];
x q80[2];
h q80[2];
x q80[3];
h q80[3];
x q80[4];
h q80[4];
x q80[5];
h q80[5];
x q80[6];
h q80[6];
x q80[7];
h q80[7];
x q80[8];
h q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722897440192 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[14], q80[12], q80[15];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722949714512 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
h q80[0];
x q80[0];
h q80[1];
x q80[1];
h q80[2];
x q80[2];
h q80[3];
x q80[3];
h q80[4];
x q80[4];
h q80[5];
x q80[5];
h q80[6];
x q80[6];
h q80[7];
x q80[7];
h q80[8];
x q80[8];
z q80[9];
mcx_gray q80[0], q80[1], q80[2], q80[3], q80[4], q80[5], q80[6], q80[7], q80[8], q80[9];
z q80[9];
x q80[0];
h q80[0];
x q80[1];
h q80[1];
x q80[2];
h q80[2];
x q80[3];
h q80[3];
x q80[4];
h q80[4];
x q80[5];
h q80[5];
x q80[6];
h q80[6];
x q80[7];
h q80[7];
x q80[8];
h q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129722949650656 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[14], q80[12], q80[15];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
ccx q80[2], q80[3], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[2], q80[3], q80[10];
cx q80[4], q80[8];
x q80[8];
cx q80[5], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[5], q80[9];
x q80[8];
cx q80[4], q80[8];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[4], q80[5], q80[10];
mcx_129717054408928 q80[10], q80[11], q80[12], q80[13], q80[14];
ccx q80[4], q80[5], q80[10];
cx q80[2], q80[8];
x q80[8];
cx q80[3], q80[9];
x q80[9];
ccx q80[8], q80[9], q80[11];
x q80[9];
cx q80[3], q80[9];
x q80[8];
cx q80[2], q80[8];
ccx q80[0], q80[1], q80[11];
ccx q80[6], q80[7], q80[10];
ccx q80[10], q80[11], q80[12];
ccx q80[6], q80[7], q80[10];
ccx q80[0], q80[1], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
ccx q80[8], q80[9], q80[11];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
mcx q80[10], q80[11], q80[12], q80[13];
cx q80[2], q80[4];
x q80[4];
cx q80[3], q80[5];
x q80[5];
ccx q80[4], q80[5], q80[10];
x q80[5];
cx q80[3], q80[5];
x q80[4];
cx q80[2], q80[4];
ccx q80[8], q80[9], q80[11];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
ccx q80[10], q80[11], q80[12];
cx q80[4], q80[6];
x q80[6];
cx q80[5], q80[7];
x q80[7];
ccx q80[6], q80[7], q80[10];
x q80[7];
cx q80[5], q80[7];
x q80[6];
cx q80[4], q80[6];
cx q80[0], q80[2];
x q80[2];
cx q80[1], q80[3];
x q80[3];
ccx q80[2], q80[3], q80[11];
x q80[3];
cx q80[1], q80[3];
x q80[2];
cx q80[0], q80[2];
h q80[0];
x q80[0];
h q80[1];
x q80[1];
h q80[2];
x q80[2];
h q80[3];
x q80[3];
h q80[4];
x q80[4];
h q80[5];
x q80[5];
h q80[6];
x q80[6];
h q80[7];
x q80[7];
h q80[8];
x q80[8];
z q80[9];
mcx_gray q80[0], q80[1], q80[2], q80[3], q80[4], q80[5], q80[6], q80[7], q80[8], q80[9];
z q80[9];
x q80[0];
h q80[0];
x q80[1];
h q80[1];
x q80[2];
h q80[2];
x q80[3];
h q80[3];
x q80[4];
h q80[4];
x q80[5];
h q80[5];
x q80[6];
h q80[6];
x q80[7];
h q80[7];
x q80[8];
h q80[8];
_creg[0] = measure q80[0];
_creg[1] = measure q80[1];
_creg[2] = measure q80[2];
_creg[3] = measure q80[3];
_creg[4] = measure q80[4];
_creg[5] = measure q80[5];
_creg[6] = measure q80[6];
_creg[7] = measure q80[7];
_creg[8] = measure q80[8];
_creg[9] = measure q80[9];
