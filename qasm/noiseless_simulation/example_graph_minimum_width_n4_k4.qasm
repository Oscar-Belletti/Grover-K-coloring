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
gate cu1_129722949983680(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
}
gate cu1_129722957960720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722957961008(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722957959472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722957960768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722957962592(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722957962400(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949994384(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949996304(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949992704(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949993232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949993376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949994000(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949944416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949944800(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949945328(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949945472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949946288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949943456(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950320336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950322208(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950321968(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950322064(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950322304(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950321824(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950323600(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950321056(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949829344(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949829584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949830832(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949829632(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949832128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949830640(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949832032(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949830112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918195888(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918196848(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918196032(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918197712(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918197136(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918198000(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918197232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918198336(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918198576(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918195264(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722918195696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901022320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901021936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901023712(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901021552(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901023664(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901021840(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901023472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901023376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901024048(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901023280(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722901022512(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950196816(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950194032(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950195952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950196624(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950193312(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950196864(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722950193216(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate mcu1_129722949983104(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  cu1_129722949983680(pi/64) _gate_q_6, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129722957960720(-pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129722949983680(pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129722957961008(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129722949983680(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129722957959472(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129722949983680(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129722957960768(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949983680(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129722957962592(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949983680(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129722957962400(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949983680(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129722949994384(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949983680(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129722949996304(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722949992704(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722949993232(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722949993376(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129722949994000(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722949944416(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722949944800(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722949945328(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949983680(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722949945472(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949946288(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722949943456(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722950320336(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722950322208(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722950321968(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722950322064(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722950322304(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722950321824(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722950323600(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722950321056(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949829344(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722949829584(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949830832(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722949829632(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949832128(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949983680(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722949830640(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949832032(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949830112(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722918195888(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722918196848(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722918196032(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722918197712(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722918197136(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129722918198000(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722918197232(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722918198336(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722918198576(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722918195264(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722918195696(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722901022320(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722901021936(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722901023712(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722901021552(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722901023664(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722901021840(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722901023472(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722901023376(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722901024048(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722901023280(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129722901022512(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722950196816(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722950194032(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722950195952(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722950196624(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722950193312(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722950196864(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722950193216(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949983680(pi/64) _gate_q_0, _gate_q_7;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  h _gate_q_7;
  mcu1_129722949983104(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7;
  h _gate_q_7;
}
bit[8] _creg;
qubit[13] q78;
h q78[0];
h q78[1];
h q78[2];
h q78[3];
h q78[4];
h q78[5];
h q78[6];
h q78[7];
x q78[12];
h q78[12];
x q78[8];
x q78[9];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
mcx q78[8], q78[9], q78[10], q78[11];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
ccx q78[8], q78[9], q78[10];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
ccx q78[11], q78[10], q78[12];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
ccx q78[8], q78[9], q78[10];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
mcx q78[8], q78[9], q78[10], q78[11];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
h q78[0];
x q78[0];
h q78[1];
x q78[1];
h q78[2];
x q78[2];
h q78[3];
x q78[3];
h q78[4];
x q78[4];
h q78[5];
x q78[5];
h q78[6];
x q78[6];
z q78[7];
mcx_gray q78[0], q78[1], q78[2], q78[3], q78[4], q78[5], q78[6], q78[7];
z q78[7];
x q78[0];
h q78[0];
x q78[1];
h q78[1];
x q78[2];
h q78[2];
x q78[3];
h q78[3];
x q78[4];
h q78[4];
x q78[5];
h q78[5];
x q78[6];
h q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
mcx q78[8], q78[9], q78[10], q78[11];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
ccx q78[8], q78[9], q78[10];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
ccx q78[11], q78[10], q78[12];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
ccx q78[8], q78[9], q78[10];
cx q78[2], q78[4];
x q78[4];
cx q78[3], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[8];
x q78[5];
cx q78[3], q78[5];
x q78[4];
cx q78[2], q78[4];
cx q78[0], q78[6];
x q78[6];
cx q78[1], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[9];
x q78[7];
cx q78[1], q78[7];
x q78[6];
cx q78[0], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
mcx q78[8], q78[9], q78[10], q78[11];
cx q78[2], q78[6];
x q78[6];
cx q78[3], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[3], q78[7];
x q78[6];
cx q78[2], q78[6];
cx q78[0], q78[4];
x q78[4];
cx q78[1], q78[5];
x q78[5];
ccx q78[4], q78[5], q78[9];
x q78[5];
cx q78[1], q78[5];
x q78[4];
cx q78[0], q78[4];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
ccx q78[8], q78[9], q78[10];
cx q78[4], q78[6];
x q78[6];
cx q78[5], q78[7];
x q78[7];
ccx q78[6], q78[7], q78[8];
x q78[7];
cx q78[5], q78[7];
x q78[6];
cx q78[4], q78[6];
cx q78[0], q78[2];
x q78[2];
cx q78[1], q78[3];
x q78[3];
ccx q78[2], q78[3], q78[9];
x q78[3];
cx q78[1], q78[3];
x q78[2];
cx q78[0], q78[2];
h q78[0];
x q78[0];
h q78[1];
x q78[1];
h q78[2];
x q78[2];
h q78[3];
x q78[3];
h q78[4];
x q78[4];
h q78[5];
x q78[5];
h q78[6];
x q78[6];
z q78[7];
mcx_gray q78[0], q78[1], q78[2], q78[3], q78[4], q78[5], q78[6], q78[7];
z q78[7];
x q78[0];
h q78[0];
x q78[1];
h q78[1];
x q78[2];
h q78[2];
x q78[3];
h q78[3];
x q78[4];
h q78[4];
x q78[5];
h q78[5];
x q78[6];
h q78[6];
_creg[0] = measure q78[0];
_creg[1] = measure q78[1];
_creg[2] = measure q78[2];
_creg[3] = measure q78[3];
_creg[4] = measure q78[4];
_creg[5] = measure q78[5];
_creg[6] = measure q78[6];
_creg[7] = measure q78[7];
