       �K"	  �bX��Abrain.Event:2ʔ�.j�     (�	�v�bX��A"��
s
A2S/observationsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
n
A2S/actionsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
q
A2S/advantagesPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
V
A2S/learning_ratePlaceholder*
dtype0*
_output_shapes
:*
shape:
Y
A2S/last_mean_policyPlaceholder*
_output_shapes
:*
shape:*
dtype0
\
A2S/last_std_dev_policyPlaceholder*
dtype0*
_output_shapes
:*
shape:
n
A2S/returnsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
W
A2S/average_rewardPlaceholder*
dtype0*
_output_shapes
:*
shape:
�
XA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  �?
�
`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shape*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
seed2*
dtype0*
_output_shapes

:@*

seed
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
�
RA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@*
T0
�
7A2S/current_policy_network/current_policy_network/fc0/w
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
>A2S/current_policy_network/current_policy_network/fc0/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/wRA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
<A2S/current_policy_network/current_policy_network/fc0/w/readIdentity7A2S/current_policy_network/current_policy_network/fc0/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
�
IA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
7A2S/current_policy_network/current_policy_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@
�
>A2S/current_policy_network/current_policy_network/fc0/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/bIA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
<A2S/current_policy_network/current_policy_network/fc0/b/readIdentity7A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
!A2S/current_policy_network/MatMulMatMulA2S/observations<A2S/current_policy_network/current_policy_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_policy_network/addAdd!A2S/current_policy_network/MatMul<A2S/current_policy_network/current_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������@
y
A2S/current_policy_network/TanhTanhA2S/current_policy_network/add*'
_output_shapes
:���������@*
T0
�
XA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes

:@@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
RA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
�
7A2S/current_policy_network/current_policy_network/fc1/w
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container 
�
>A2S/current_policy_network/current_policy_network/fc1/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/wRA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
<A2S/current_policy_network/current_policy_network/fc1/w/readIdentity7A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
IA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
7A2S/current_policy_network/current_policy_network/fc1/b
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
>A2S/current_policy_network/current_policy_network/fc1/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/bIA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zeros*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
<A2S/current_policy_network/current_policy_network/fc1/b/readIdentity7A2S/current_policy_network/current_policy_network/fc1/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
�
#A2S/current_policy_network/MatMul_1MatMulA2S/current_policy_network/Tanh<A2S/current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
 A2S/current_policy_network/add_1Add#A2S/current_policy_network/MatMul_1<A2S/current_policy_network/current_policy_network/fc1/b/read*
T0*'
_output_shapes
:���������@
}
!A2S/current_policy_network/Tanh_1Tanh A2S/current_policy_network/add_1*
T0*'
_output_shapes
:���������@
�
XA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB"@      *
dtype0
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
seed2-
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
RA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
7A2S/current_policy_network/current_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@
�
>A2S/current_policy_network/current_policy_network/out/w/AssignAssign7A2S/current_policy_network/current_policy_network/out/wRA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
<A2S/current_policy_network/current_policy_network/out/w/readIdentity7A2S/current_policy_network/current_policy_network/out/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
IA2S/current_policy_network/current_policy_network/out/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
7A2S/current_policy_network/current_policy_network/out/b
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
>A2S/current_policy_network/current_policy_network/out/b/AssignAssign7A2S/current_policy_network/current_policy_network/out/bIA2S/current_policy_network/current_policy_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
�
<A2S/current_policy_network/current_policy_network/out/b/readIdentity7A2S/current_policy_network/current_policy_network/out/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
�
#A2S/current_policy_network/MatMul_2MatMul!A2S/current_policy_network/Tanh_1<A2S/current_policy_network/current_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
 A2S/current_policy_network/add_2Add#A2S/current_policy_network/MatMul_2<A2S/current_policy_network/current_policy_network/out/b/read*
T0*'
_output_shapes
:���������
�
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shape*
seed2=*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
�
1A2S/best_policy_network/best_policy_network/fc0/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container 
�
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
�
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:@
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������@
s
A2S/best_policy_network/TanhTanhA2S/best_policy_network/add*
T0*'
_output_shapes
:���������@
�
RA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shape*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
seed2N*
dtype0*
_output_shapes

:@@*

seed*
T0
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
�
LA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
�
1A2S/best_policy_network/best_policy_network/fc1/w
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
�
8A2S/best_policy_network/best_policy_network/fc1/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/wLA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
6A2S/best_policy_network/best_policy_network/fc1/w/readIdentity1A2S/best_policy_network/best_policy_network/fc1/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
�
CA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
1A2S/best_policy_network/best_policy_network/fc1/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
	container 
�
8A2S/best_policy_network/best_policy_network/fc1/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/bCA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
6A2S/best_policy_network/best_policy_network/fc1/b/readIdentity1A2S/best_policy_network/best_policy_network/fc1/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
_output_shapes
:@
�
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/Tanh6A2S/best_policy_network/best_policy_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/best_policy_network/add_1Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/fc1/b/read*'
_output_shapes
:���������@*
T0
w
A2S/best_policy_network/Tanh_1TanhA2S/best_policy_network/add_1*'
_output_shapes
:���������@*
T0
�
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"@      
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
ZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2_
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
�
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@*
T0
�
1A2S/best_policy_network/best_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:@
�
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
�
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
 A2S/best_policy_network/MatMul_2MatMulA2S/best_policy_network/Tanh_16A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_26A2S/best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:���������
�
RA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB"   @   
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  �?
�
ZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shape*
seed2o*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes
: 
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
�
LA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
�
1A2S/last_policy_network/last_policy_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
	container *
shape
:@
�
8A2S/last_policy_network/last_policy_network/fc0/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/wLA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w
�
6A2S/last_policy_network/last_policy_network/fc0/w/readIdentity1A2S/last_policy_network/last_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
�
CA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
1A2S/last_policy_network/last_policy_network/fc0/b
VariableV2*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
8A2S/last_policy_network/last_policy_network/fc0/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/bCA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
6A2S/last_policy_network/last_policy_network/fc0/b/readIdentity1A2S/last_policy_network/last_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
_output_shapes
:@
�
A2S/last_policy_network/MatMulMatMulA2S/observations6A2S/last_policy_network/last_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/last_policy_network/addAddA2S/last_policy_network/MatMul6A2S/last_policy_network/last_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������@
s
A2S/last_policy_network/TanhTanhA2S/last_policy_network/add*'
_output_shapes
:���������@*
T0
�
RA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB"@   @   *
dtype0
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
seed2�
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
�
LA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w
�
1A2S/last_policy_network/last_policy_network/fc1/w
VariableV2*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
8A2S/last_policy_network/last_policy_network/fc1/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/wLA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform*
_output_shapes

:@@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(
�
6A2S/last_policy_network/last_policy_network/fc1/w/readIdentity1A2S/last_policy_network/last_policy_network/fc1/w*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@*
T0
�
CA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
valueB@*    
�
1A2S/last_policy_network/last_policy_network/fc1/b
VariableV2*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
8A2S/last_policy_network/last_policy_network/fc1/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/bCA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b
�
6A2S/last_policy_network/last_policy_network/fc1/b/readIdentity1A2S/last_policy_network/last_policy_network/fc1/b*
_output_shapes
:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b
�
 A2S/last_policy_network/MatMul_1MatMulA2S/last_policy_network/Tanh6A2S/last_policy_network/last_policy_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/last_policy_network/add_1Add A2S/last_policy_network/MatMul_16A2S/last_policy_network/last_policy_network/fc1/b/read*
T0*'
_output_shapes
:���������@
w
A2S/last_policy_network/Tanh_1TanhA2S/last_policy_network/add_1*'
_output_shapes
:���������@*
T0
�
RA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB"@      
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *��̽
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
ZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
seed2�
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes
: 
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
�
LA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
�
1A2S/last_policy_network/last_policy_network/out/w
VariableV2*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
8A2S/last_policy_network/last_policy_network/out/w/AssignAssign1A2S/last_policy_network/last_policy_network/out/wLA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
�
6A2S/last_policy_network/last_policy_network/out/w/readIdentity1A2S/last_policy_network/last_policy_network/out/w*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@*
T0
�
CA2S/last_policy_network/last_policy_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
valueB*    
�
1A2S/last_policy_network/last_policy_network/out/b
VariableV2*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
8A2S/last_policy_network/last_policy_network/out/b/AssignAssign1A2S/last_policy_network/last_policy_network/out/bCA2S/last_policy_network/last_policy_network/out/b/Initializer/zeros*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
6A2S/last_policy_network/last_policy_network/out/b/readIdentity1A2S/last_policy_network/last_policy_network/out/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
_output_shapes
:
�
 A2S/last_policy_network/MatMul_2MatMulA2S/last_policy_network/Tanh_16A2S/last_policy_network/last_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/last_policy_network/add_2Add A2S/last_policy_network/MatMul_26A2S/last_policy_network/last_policy_network/out/b/read*'
_output_shapes
:���������*
T0
�
VA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB"   @   
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
seed2�
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
�
PA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
�
5A2S/current_value_network/current_value_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
	container *
shape
:@
�
<A2S/current_value_network/current_value_network/fc0/w/AssignAssign5A2S/current_value_network/current_value_network/fc0/wPA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
:A2S/current_value_network/current_value_network/fc0/w/readIdentity5A2S/current_value_network/current_value_network/fc0/w*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
�
GA2S/current_value_network/current_value_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
5A2S/current_value_network/current_value_network/fc0/b
VariableV2*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
<A2S/current_value_network/current_value_network/fc0/b/AssignAssign5A2S/current_value_network/current_value_network/fc0/bGA2S/current_value_network/current_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
:A2S/current_value_network/current_value_network/fc0/b/readIdentity5A2S/current_value_network/current_value_network/fc0/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
�
 A2S/current_value_network/MatMulMatMulA2S/observations:A2S/current_value_network/current_value_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_value_network/addAdd A2S/current_value_network/MatMul:A2S/current_value_network/current_value_network/fc0/b/read*'
_output_shapes
:���������@*
T0
w
A2S/current_value_network/TanhTanhA2S/current_value_network/add*
T0*'
_output_shapes
:���������@
�
VA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shape*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
seed2�*
dtype0*
_output_shapes

:@@
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/sub*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0
�
PA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0
�
5A2S/current_value_network/current_value_network/fc1/w
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
<A2S/current_value_network/current_value_network/fc1/w/AssignAssign5A2S/current_value_network/current_value_network/fc1/wPA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
:A2S/current_value_network/current_value_network/fc1/w/readIdentity5A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
GA2S/current_value_network/current_value_network/fc1/b/Initializer/zerosConst*
_output_shapes
:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0
�
5A2S/current_value_network/current_value_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@
�
<A2S/current_value_network/current_value_network/fc1/b/AssignAssign5A2S/current_value_network/current_value_network/fc1/bGA2S/current_value_network/current_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
:A2S/current_value_network/current_value_network/fc1/b/readIdentity5A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
"A2S/current_value_network/MatMul_1MatMulA2S/current_value_network/Tanh:A2S/current_value_network/current_value_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_value_network/add_1Add"A2S/current_value_network/MatMul_1:A2S/current_value_network/current_value_network/fc1/b/read*'
_output_shapes
:���������@*
T0
{
 A2S/current_value_network/Tanh_1TanhA2S/current_value_network/add_1*
T0*'
_output_shapes
:���������@
�
VA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB"@      *
dtype0
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shape*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
seed2�*
dtype0*
_output_shapes

:@*

seed*
T0
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
�
PA2S/current_value_network/current_value_network/out/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
�
5A2S/current_value_network/current_value_network/out/w
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
<A2S/current_value_network/current_value_network/out/w/AssignAssign5A2S/current_value_network/current_value_network/out/wPA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(
�
:A2S/current_value_network/current_value_network/out/w/readIdentity5A2S/current_value_network/current_value_network/out/w*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
�
GA2S/current_value_network/current_value_network/out/b/Initializer/zerosConst*
_output_shapes
:*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0
�
5A2S/current_value_network/current_value_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container *
shape:
�
<A2S/current_value_network/current_value_network/out/b/AssignAssign5A2S/current_value_network/current_value_network/out/bGA2S/current_value_network/current_value_network/out/b/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
:A2S/current_value_network/current_value_network/out/b/readIdentity5A2S/current_value_network/current_value_network/out/b*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:*
T0
�
"A2S/current_value_network/MatMul_2MatMul A2S/current_value_network/Tanh_1:A2S/current_value_network/current_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/current_value_network/add_2Add"A2S/current_value_network/MatMul_2:A2S/current_value_network/current_value_network/out/b/read*
T0*'
_output_shapes
:���������
�
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:@*

seed
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@
�
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@
�
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
/A2S/best_value_network/best_value_network/fc0/b
VariableV2*
_output_shapes
:@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:@*
dtype0
�
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(
�
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
�
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*'
_output_shapes
:���������@*
T0
q
A2S/best_value_network/TanhTanhA2S/best_value_network/add*
T0*'
_output_shapes
:���������@
�
PA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  ��
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
XA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
seed2�
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
�
JA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
/A2S/best_value_network/best_value_network/fc1/w
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
6A2S/best_value_network/best_value_network/fc1/w/AssignAssign/A2S/best_value_network/best_value_network/fc1/wJA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
4A2S/best_value_network/best_value_network/fc1/w/readIdentity/A2S/best_value_network/best_value_network/fc1/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
�
AA2S/best_value_network/best_value_network/fc1/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
/A2S/best_value_network/best_value_network/fc1/b
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
6A2S/best_value_network/best_value_network/fc1/b/AssignAssign/A2S/best_value_network/best_value_network/fc1/bAA2S/best_value_network/best_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
4A2S/best_value_network/best_value_network/fc1/b/readIdentity/A2S/best_value_network/best_value_network/fc1/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
_output_shapes
:@
�
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/Tanh4A2S/best_value_network/best_value_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/best_value_network/add_1AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/fc1/b/read*
T0*'
_output_shapes
:���������@
u
A2S/best_value_network/Tanh_1TanhA2S/best_value_network/add_1*'
_output_shapes
:���������@*
T0
�
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *��̽*
dtype0
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
XA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2�
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@
�
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@
�
/A2S/best_value_network/best_value_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:@
�
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@
�
AA2S/best_value_network/best_value_network/out/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
/A2S/best_value_network/best_value_network/out/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container 
�
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
�
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
�
A2S/best_value_network/MatMul_2MatMulA2S/best_value_network/Tanh_14A2S/best_value_network/best_value_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_24A2S/best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:���������
h
A2S/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
j
A2S/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
j
A2S/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_sliceStridedSlice A2S/current_policy_network/add_2A2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
`
A2S/SqueezeSqueezeA2S/strided_slice*
squeeze_dims
 *
T0*
_output_shapes
:
b
A2S/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
v
A2S/ReshapeReshapeA2S/SqueezeA2S/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
j
A2S/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
l
A2S/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_slice_1StridedSliceA2S/best_policy_network/add_2A2S/strided_slice_1/stackA2S/strided_slice_1/stack_1A2S/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
d
A2S/Squeeze_1SqueezeA2S/strided_slice_1*
_output_shapes
:*
squeeze_dims
 *
T0
d
A2S/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
|
A2S/Reshape_1ReshapeA2S/Squeeze_1A2S/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:
l
A2S/strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
A2S/strided_slice_2StridedSliceA2S/last_policy_network/add_2A2S/strided_slice_2/stackA2S/strided_slice_2/stack_1A2S/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
d
A2S/Squeeze_2SqueezeA2S/strided_slice_2*
T0*
_output_shapes
:*
squeeze_dims
 
d
A2S/Reshape_2/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
|
A2S/Reshape_2ReshapeA2S/Squeeze_2A2S/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_3/stackConst*
_output_shapes
:*
valueB"       *
dtype0
l
A2S/strided_slice_3/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
l
A2S/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
A2S/strided_slice_3StridedSlice A2S/current_policy_network/add_2A2S/strided_slice_3/stackA2S/strided_slice_3/stack_1A2S/strided_slice_3/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
d
A2S/Squeeze_3SqueezeA2S/strided_slice_3*
squeeze_dims
 *
T0*
_output_shapes
:
J
A2S/SoftplusSoftplusA2S/Squeeze_3*
T0*
_output_shapes
:
N
	A2S/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *��'7
J
A2S/addAddA2S/Softplus	A2S/add/y*
T0*
_output_shapes
:
d
A2S/Reshape_3/shapeConst*
_output_shapes
:*
valueB"����   *
dtype0
v
A2S/Reshape_3ReshapeA2S/addA2S/Reshape_3/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_4/stackConst*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
l
A2S/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_slice_4StridedSliceA2S/best_policy_network/add_2A2S/strided_slice_4/stackA2S/strided_slice_4/stack_1A2S/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
d
A2S/Squeeze_4SqueezeA2S/strided_slice_4*
squeeze_dims
 *
T0*
_output_shapes
:
L
A2S/Softplus_1SoftplusA2S/Squeeze_4*
_output_shapes
:*
T0
P
A2S/add_1/yConst*
valueB
 *��'7*
dtype0*
_output_shapes
: 
P
	A2S/add_1AddA2S/Softplus_1A2S/add_1/y*
T0*
_output_shapes
:
d
A2S/Reshape_4/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
x
A2S/Reshape_4Reshape	A2S/add_1A2S/Reshape_4/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_5/stackConst*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_5/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
l
A2S/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_slice_5StridedSliceA2S/last_policy_network/add_2A2S/strided_slice_5/stackA2S/strided_slice_5/stack_1A2S/strided_slice_5/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
Index0*
T0
d
A2S/Squeeze_5SqueezeA2S/strided_slice_5*
squeeze_dims
 *
T0*
_output_shapes
:
L
A2S/Softplus_2SoftplusA2S/Squeeze_5*
_output_shapes
:*
T0
P
A2S/add_2/yConst*
valueB
 *��'7*
dtype0*
_output_shapes
: 
P
	A2S/add_2AddA2S/Softplus_2A2S/add_2/y*
T0*
_output_shapes
:
d
A2S/Reshape_5/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
x
A2S/Reshape_5Reshape	A2S/add_2A2S/Reshape_5/shape*
T0*
Tshape0*'
_output_shapes
:���������
Y
A2S/Normal/locIdentityA2S/Reshape*
T0*'
_output_shapes
:���������
]
A2S/Normal/scaleIdentityA2S/Reshape_3*
T0*'
_output_shapes
:���������
]
A2S/Normal_1/locIdentityA2S/Reshape_1*
T0*'
_output_shapes
:���������
_
A2S/Normal_1/scaleIdentityA2S/Reshape_4*'
_output_shapes
:���������*
T0
]
A2S/Normal_2/locIdentityA2S/Reshape_2*'
_output_shapes
:���������*
T0
_
A2S/Normal_2/scaleIdentityA2S/Reshape_5*
T0*'
_output_shapes
:���������
o
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
q
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
q
,A2S/KullbackLeibler/kl_normal_normal/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
y
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal/scale*
T0*'
_output_shapes
:���������
}
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal_2/scale*'
_output_shapes
:���������*
T0
�
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*'
_output_shapes
:���������
�
(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal/locA2S/Normal_2/loc*'
_output_shapes
:���������*
T0
�
-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:���������
�
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*'
_output_shapes
:���������
�
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:���������
�
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*'
_output_shapes
:���������*
T0
�
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
T0*'
_output_shapes
:���������
�
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
T0*'
_output_shapes
:���������
�
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
T0*'
_output_shapes
:���������
�
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*'
_output_shapes
:���������*
T0
Z
	A2S/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/add	A2S/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
A2S/kl/tagsConst*
dtype0*
_output_shapes
: *
valueB BA2S/kl
O
A2S/klScalarSummaryA2S/kl/tagsA2S/Mean*
_output_shapes
: *
T0
s
%A2S/Normal_3/batch_shape_tensor/ShapeShapeA2S/Normal/loc*
out_type0*
_output_shapes
:*
T0
w
'A2S/Normal_3/batch_shape_tensor/Shape_1ShapeA2S/Normal/scale*
T0*
out_type0*
_output_shapes
:
�
-A2S/Normal_3/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_3/batch_shape_tensor/Shape'A2S/Normal_3/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
]
A2S/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
Q
A2S/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_3/batch_shape_tensor/BroadcastArgsA2S/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
[
A2S/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
]
A2S/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :������������������*
seed2�*

seed
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :������������������
i
A2S/mulMulA2S/random_normalA2S/Normal/scale*
T0*+
_output_shapes
:���������
_
	A2S/add_3AddA2S/mulA2S/Normal/loc*
T0*+
_output_shapes
:���������
h
A2S/Reshape_6/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����      
|
A2S/Reshape_6Reshape	A2S/add_3A2S/Reshape_6/shape*
T0*
Tshape0*+
_output_shapes
:���������
S
A2S/concat_1/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*
N*'
_output_shapes
:���������*

Tidx0*
T0
�
NA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  ��
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  �?
�
VA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
seed2�
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
�
HA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
�
-A2S/current_q_network/current_q_network/fc0/w
VariableV2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
4A2S/current_q_network/current_q_network/fc0/w/AssignAssign-A2S/current_q_network/current_q_network/fc0/wHA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
2A2S/current_q_network/current_q_network/fc0/w/readIdentity-A2S/current_q_network/current_q_network/fc0/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
�
?A2S/current_q_network/current_q_network/fc0/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
-A2S/current_q_network/current_q_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape:@
�
4A2S/current_q_network/current_q_network/fc0/b/AssignAssign-A2S/current_q_network/current_q_network/fc0/b?A2S/current_q_network/current_q_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
2A2S/current_q_network/current_q_network/fc0/b/readIdentity-A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
A2S/current_q_network/MatMulMatMulA2S/concat_12A2S/current_q_network/current_q_network/fc0/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/current_q_network/addAddA2S/current_q_network/MatMul2A2S/current_q_network/current_q_network/fc0/b/read*'
_output_shapes
:���������@*
T0
o
A2S/current_q_network/TanhTanhA2S/current_q_network/add*
T0*'
_output_shapes
:���������@
�
NA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
VA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
seed2�
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
�
HA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
�
-A2S/current_q_network/current_q_network/fc1/w
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container 
�
4A2S/current_q_network/current_q_network/fc1/w/AssignAssign-A2S/current_q_network/current_q_network/fc1/wHA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
�
2A2S/current_q_network/current_q_network/fc1/w/readIdentity-A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
�
?A2S/current_q_network/current_q_network/fc1/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
-A2S/current_q_network/current_q_network/fc1/b
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
4A2S/current_q_network/current_q_network/fc1/b/AssignAssign-A2S/current_q_network/current_q_network/fc1/b?A2S/current_q_network/current_q_network/fc1/b/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
2A2S/current_q_network/current_q_network/fc1/b/readIdentity-A2S/current_q_network/current_q_network/fc1/b*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@*
T0
�
A2S/current_q_network/MatMul_1MatMulA2S/current_q_network/Tanh2A2S/current_q_network/current_q_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_q_network/add_1AddA2S/current_q_network/MatMul_12A2S/current_q_network/current_q_network/fc1/b/read*'
_output_shapes
:���������@*
T0
s
A2S/current_q_network/Tanh_1TanhA2S/current_q_network/add_1*'
_output_shapes
:���������@*
T0
�
NA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *���=*
dtype0
�
VA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shape*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
seed2�*
dtype0
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
HA2S/current_q_network/current_q_network/out/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@*
T0
�
-A2S/current_q_network/current_q_network/out/w
VariableV2*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
	container *
shape
:@*
dtype0
�
4A2S/current_q_network/current_q_network/out/w/AssignAssign-A2S/current_q_network/current_q_network/out/wHA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(
�
2A2S/current_q_network/current_q_network/out/w/readIdentity-A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w
�
?A2S/current_q_network/current_q_network/out/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
-A2S/current_q_network/current_q_network/out/b
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
4A2S/current_q_network/current_q_network/out/b/AssignAssign-A2S/current_q_network/current_q_network/out/b?A2S/current_q_network/current_q_network/out/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
�
2A2S/current_q_network/current_q_network/out/b/readIdentity-A2S/current_q_network/current_q_network/out/b*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:
�
A2S/current_q_network/MatMul_2MatMulA2S/current_q_network/Tanh_12A2S/current_q_network/current_q_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/current_q_network/add_2AddA2S/current_q_network/MatMul_22A2S/current_q_network/current_q_network/out/b/read*
T0*'
_output_shapes
:���������
�
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:@*

seed
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@
�
'A2S/best_q_network/best_q_network/fc0/w
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@*
T0
�
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB@*    
�
'A2S/best_q_network/best_q_network/fc0/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container 
�
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:@
�
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*
T0*'
_output_shapes
:���������@
i
A2S/best_q_network/TanhTanhA2S/best_q_network/add*
T0*'
_output_shapes
:���������@
�
HA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  �?
�
PA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
seed2�
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
�
BA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w
�
'A2S/best_q_network/best_q_network/fc1/w
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w
�
.A2S/best_q_network/best_q_network/fc1/w/AssignAssign'A2S/best_q_network/best_q_network/fc1/wBA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
,A2S/best_q_network/best_q_network/fc1/w/readIdentity'A2S/best_q_network/best_q_network/fc1/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
�
9A2S/best_q_network/best_q_network/fc1/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
'A2S/best_q_network/best_q_network/fc1/b
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
.A2S/best_q_network/best_q_network/fc1/b/AssignAssign'A2S/best_q_network/best_q_network/fc1/b9A2S/best_q_network/best_q_network/fc1/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
,A2S/best_q_network/best_q_network/fc1/b/readIdentity'A2S/best_q_network/best_q_network/fc1/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
_output_shapes
:@
�
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/Tanh,A2S/best_q_network/best_q_network/fc1/w/read*
transpose_b( *
T0*'
_output_shapes
:���������@*
transpose_a( 
�
A2S/best_q_network/add_1AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/fc1/b/read*
T0*'
_output_shapes
:���������@
m
A2S/best_q_network/Tanh_1TanhA2S/best_q_network/add_1*
T0*'
_output_shapes
:���������@
�
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *��̽*
dtype0
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
seed2�
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@
�
'A2S/best_q_network/best_q_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:@
�
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@*
T0
�
9A2S/best_q_network/best_q_network/out/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
'A2S/best_q_network/best_q_network/out/b
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
�
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
�
A2S/best_q_network/MatMul_2MatMulA2S/best_q_network/Tanh_1,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_2,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:���������
{
%A2S/Normal_4/log_prob/standardize/subSubA2S/actionsA2S/Normal/loc*
T0*'
_output_shapes
:���������
�
)A2S/Normal_4/log_prob/standardize/truedivRealDiv%A2S/Normal_4/log_prob/standardize/subA2S/Normal/scale*
T0*'
_output_shapes
:���������
�
A2S/Normal_4/log_prob/SquareSquare)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
`
A2S/Normal_4/log_prob/mul/xConst*
valueB
 *   �*
dtype0*
_output_shapes
: 
�
A2S/Normal_4/log_prob/mulMulA2S/Normal_4/log_prob/mul/xA2S/Normal_4/log_prob/Square*
T0*'
_output_shapes
:���������
d
A2S/Normal_4/log_prob/LogLogA2S/Normal/scale*
T0*'
_output_shapes
:���������
`
A2S/Normal_4/log_prob/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *�?k?
�
A2S/Normal_4/log_prob/addAddA2S/Normal_4/log_prob/add/xA2S/Normal_4/log_prob/Log*
T0*'
_output_shapes
:���������
�
A2S/Normal_4/log_prob/subSubA2S/Normal_4/log_prob/mulA2S/Normal_4/log_prob/add*'
_output_shapes
:���������*
T0
[
A2S/NegNegA2S/Normal_4/log_prob/sub*
T0*'
_output_shapes
:���������
[
	A2S/mul_1MulA2S/NegA2S/advantages*'
_output_shapes
:���������*
T0
\
A2S/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
h

A2S/Mean_1Mean	A2S/mul_1A2S/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
t
A2S/policy_network_loss/tagsConst*(
valueB BA2S/policy_network_loss*
dtype0*
_output_shapes
: 
s
A2S/policy_network_lossScalarSummaryA2S/policy_network_loss/tags
A2S/Mean_1*
T0*
_output_shapes
: 
�
A2S/SquaredDifferenceSquaredDifferenceA2S/current_value_network/add_2A2S/returns*
T0*'
_output_shapes
:���������
\
A2S/Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
t

A2S/Mean_2MeanA2S/SquaredDifferenceA2S/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
r
A2S/value_network_loss/tagsConst*'
valueB BA2S/value_network_loss*
dtype0*
_output_shapes
: 
q
A2S/value_network_lossScalarSummaryA2S/value_network_loss/tags
A2S/Mean_2*
T0*
_output_shapes
: 
�
A2S/SquaredDifference_1SquaredDifferenceA2S/current_q_network/add_2A2S/returns*
T0*'
_output_shapes
:���������
\
A2S/Const_3Const*
_output_shapes
:*
valueB"       *
dtype0
v

A2S/Mean_3MeanA2S/SquaredDifference_1A2S/Const_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
A2S/q_network_loss/tagsConst*#
valueB BA2S/q_network_loss*
dtype0*
_output_shapes
: 
i
A2S/q_network_lossScalarSummaryA2S/q_network_loss/tags
A2S/Mean_3*
_output_shapes
: *
T0
V
A2S/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
A2S/gradients/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
e
A2S/gradients/FillFillA2S/gradients/ShapeA2S/gradients/Const*
T0*
_output_shapes
: 
|
+A2S/gradients/A2S/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
%A2S/gradients/A2S/Mean_1_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
#A2S/gradients/A2S/Mean_1_grad/ShapeShape	A2S/mul_1*
_output_shapes
:*
T0*
out_type0
�
"A2S/gradients/A2S/Mean_1_grad/TileTile%A2S/gradients/A2S/Mean_1_grad/Reshape#A2S/gradients/A2S/Mean_1_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
n
%A2S/gradients/A2S/Mean_1_grad/Shape_1Shape	A2S/mul_1*
T0*
out_type0*
_output_shapes
:
h
%A2S/gradients/A2S/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
m
#A2S/gradients/A2S/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"A2S/gradients/A2S/Mean_1_grad/ProdProd%A2S/gradients/A2S/Mean_1_grad/Shape_1#A2S/gradients/A2S/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
o
%A2S/gradients/A2S/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients/A2S/Mean_1_grad/Prod_1Prod%A2S/gradients/A2S/Mean_1_grad/Shape_2%A2S/gradients/A2S/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
'A2S/gradients/A2S/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
%A2S/gradients/A2S/Mean_1_grad/MaximumMaximum$A2S/gradients/A2S/Mean_1_grad/Prod_1'A2S/gradients/A2S/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
�
&A2S/gradients/A2S/Mean_1_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_1_grad/Prod%A2S/gradients/A2S/Mean_1_grad/Maximum*
_output_shapes
: *
T0
�
"A2S/gradients/A2S/Mean_1_grad/CastCast&A2S/gradients/A2S/Mean_1_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
%A2S/gradients/A2S/Mean_1_grad/truedivRealDiv"A2S/gradients/A2S/Mean_1_grad/Tile"A2S/gradients/A2S/Mean_1_grad/Cast*
T0*'
_output_shapes
:���������
i
"A2S/gradients/A2S/mul_1_grad/ShapeShapeA2S/Neg*
_output_shapes
:*
T0*
out_type0
r
$A2S/gradients/A2S/mul_1_grad/Shape_1ShapeA2S/advantages*
T0*
out_type0*
_output_shapes
:
�
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_1_grad/truedivA2S/advantages*
T0*'
_output_shapes
:���������
�
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_1_grad/truediv*
T0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1
�
5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape
�
7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ShapeShapeA2S/Normal_4/log_prob/mul*
T0*
out_type0*
_output_shapes
:
�
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1ShapeA2S/Normal_4/log_prob/add*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
�
6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape
�
GA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1*'
_output_shapes
:���������
u
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1ShapeA2S/Normal_4/log_prob/Square*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_4/log_prob/Square*'
_output_shapes
:���������*
T0
�
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1MulA2S/Normal_4/log_prob/mul/xEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape*
_output_shapes
: 
�
GA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1*'
_output_shapes
:���������
u
2A2S/gradients/A2S/Normal_4/log_prob/add_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
4A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape_1ShapeA2S/Normal_4/log_prob/Log*
out_type0*
_output_shapes
:*
T0
�
BA2S/gradients/A2S/Normal_4/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/add_grad/SumSumGA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1BA2S/gradients/A2S/Normal_4/log_prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients/A2S/Normal_4/log_prob/add_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/add_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Sum_1SumGA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1DA2S/gradients/A2S/Normal_4/log_prob/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Sum_14A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape*
_output_shapes
: 
�
GA2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
3A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/x)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul*'
_output_shapes
:���������*
T0
�
7A2S/gradients/A2S/Normal_4/log_prob/Log_grad/Reciprocal
ReciprocalA2S/Normal/scaleH^A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/Log_grad/mulMulGA2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependency_17A2S/gradients/A2S/Normal_4/log_prob/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_4/log_prob/standardize/sub*
_output_shapes
:*
T0*
out_type0
�
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1ShapeA2S/Normal/scale*
out_type0*
_output_shapes
:*
T0
�
RA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1A2S/Normal/scale*'
_output_shapes
:���������*
T0
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_4/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegA2S/Normal/scale*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
MA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1
�
UA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:���������
�
WA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
out_type0*
_output_shapes
:*
T0
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal/loc*
T0*
out_type0*
_output_shapes
:
�
NA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
IA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1
�
QA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape
�
SA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1
�
A2S/gradients/AddNAddN0A2S/gradients/A2S/Normal_4/log_prob/Log_grad/mulWA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependency_1*
N*'
_output_shapes
:���������*
T0*C
_class9
75loc:@A2S/gradients/A2S/Normal_4/log_prob/Log_grad/mul
v
&A2S/gradients/A2S/Reshape_3_grad/ShapeShapeA2S/add*
out_type0*#
_output_shapes
:���������*
T0
�
(A2S/gradients/A2S/Reshape_3_grad/ReshapeReshapeA2S/gradients/AddN&A2S/gradients/A2S/Reshape_3_grad/Shape*
T0*
Tshape0*
_output_shapes
:
x
$A2S/gradients/A2S/Reshape_grad/ShapeShapeA2S/Squeeze*
T0*
out_type0*#
_output_shapes
:���������
�
&A2S/gradients/A2S/Reshape_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1$A2S/gradients/A2S/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
u
 A2S/gradients/A2S/add_grad/ShapeShapeA2S/Softplus*
T0*
out_type0*#
_output_shapes
:���������
e
"A2S/gradients/A2S/add_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
0A2S/gradients/A2S/add_grad/BroadcastGradientArgsBroadcastGradientArgs A2S/gradients/A2S/add_grad/Shape"A2S/gradients/A2S/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
A2S/gradients/A2S/add_grad/SumSum(A2S/gradients/A2S/Reshape_3_grad/Reshape0A2S/gradients/A2S/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
"A2S/gradients/A2S/add_grad/ReshapeReshapeA2S/gradients/A2S/add_grad/Sum A2S/gradients/A2S/add_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
 A2S/gradients/A2S/add_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_3_grad/Reshape2A2S/gradients/A2S/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
$A2S/gradients/A2S/add_grad/Reshape_1Reshape A2S/gradients/A2S/add_grad/Sum_1"A2S/gradients/A2S/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+A2S/gradients/A2S/add_grad/tuple/group_depsNoOp#^A2S/gradients/A2S/add_grad/Reshape%^A2S/gradients/A2S/add_grad/Reshape_1
�
3A2S/gradients/A2S/add_grad/tuple/control_dependencyIdentity"A2S/gradients/A2S/add_grad/Reshape,^A2S/gradients/A2S/add_grad/tuple/group_deps*5
_class+
)'loc:@A2S/gradients/A2S/add_grad/Reshape*
_output_shapes
:*
T0
�
5A2S/gradients/A2S/add_grad/tuple/control_dependency_1Identity$A2S/gradients/A2S/add_grad/Reshape_1,^A2S/gradients/A2S/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@A2S/gradients/A2S/add_grad/Reshape_1*
_output_shapes
: 
u
$A2S/gradients/A2S/Squeeze_grad/ShapeShapeA2S/strided_slice*
T0*
out_type0*
_output_shapes
:
�
&A2S/gradients/A2S/Squeeze_grad/ReshapeReshape&A2S/gradients/A2S/Reshape_grad/Reshape$A2S/gradients/A2S/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
,A2S/gradients/A2S/Softplus_grad/SoftplusGradSoftplusGrad3A2S/gradients/A2S/add_grad/tuple/control_dependencyA2S/Squeeze_3*
T0*
_output_shapes
:
�
*A2S/gradients/A2S/strided_slice_grad/ShapeShape A2S/current_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
�
5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradStridedSliceGrad*A2S/gradients/A2S/strided_slice_grad/ShapeA2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2&A2S/gradients/A2S/Squeeze_grad/Reshape*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
Index0*
T0
y
&A2S/gradients/A2S/Squeeze_3_grad/ShapeShapeA2S/strided_slice_3*
T0*
out_type0*
_output_shapes
:
�
(A2S/gradients/A2S/Squeeze_3_grad/ReshapeReshape,A2S/gradients/A2S/Softplus_grad/SoftplusGrad&A2S/gradients/A2S/Squeeze_3_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
,A2S/gradients/A2S/strided_slice_3_grad/ShapeShape A2S/current_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
�
7A2S/gradients/A2S/strided_slice_3_grad/StridedSliceGradStridedSliceGrad,A2S/gradients/A2S/strided_slice_3_grad/ShapeA2S/strided_slice_3/stackA2S/strided_slice_3/stack_1A2S/strided_slice_3/stack_2(A2S/gradients/A2S/Squeeze_3_grad/Reshape*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
�
A2S/gradients/AddN_1AddN5A2S/gradients/A2S/strided_slice_grad/StridedSliceGrad7A2S/gradients/A2S/strided_slice_3_grad/StridedSliceGrad*
T0*H
_class>
<:loc:@A2S/gradients/A2S/strided_slice_grad/StridedSliceGrad*
N*'
_output_shapes
:���������
�
9A2S/gradients/A2S/current_policy_network/add_2_grad/ShapeShape#A2S/current_policy_network/MatMul_2*
T0*
out_type0*
_output_shapes
:
�
;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
IA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
7A2S/gradients/A2S/current_policy_network/add_2_grad/SumSumA2S/gradients/AddN_1IA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_2_grad/Sum9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1SumA2S/gradients/AddN_1KA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
DA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1
�
LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
NA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
�
=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/out/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1MatMul!A2S/current_policy_network/Tanh_1LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
GA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1
�
OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul
�
QA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*
_output_shapes

:@*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1
�
=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradTanhGrad!A2S/current_policy_network/Tanh_1OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
9A2S/gradients/A2S/current_policy_network/add_1_grad/ShapeShape#A2S/current_policy_network/MatMul_1*
_output_shapes
:*
T0*
out_type0
�
;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
IA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
7A2S/gradients/A2S/current_policy_network/add_1_grad/SumSum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_1_grad/Sum9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1Sum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradKA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*
Tshape0*
_output_shapes
:@*
T0
�
DA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1
�
LA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape
�
NA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1*
_output_shapes
:@
�
=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/current_policy_network/TanhLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
GA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1
�
OA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul
�
QA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
�
;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradTanhGradA2S/current_policy_network/TanhOA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
7A2S/gradients/A2S/current_policy_network/add_grad/ShapeShape!A2S/current_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1Const*
_output_shapes
:*
valueB:@*
dtype0
�
GA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients/A2S/current_policy_network/add_grad/Shape9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
5A2S/gradients/A2S/current_policy_network/add_grad/SumSum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradGA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeReshape5A2S/gradients/A2S/current_policy_network/add_grad/Sum7A2S/gradients/A2S/current_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
7A2S/gradients/A2S/current_policy_network/add_grad/Sum_1Sum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1Reshape7A2S/gradients/A2S/current_policy_network/add_grad/Sum_19A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
BA2S/gradients/A2S/current_policy_network/add_grad/tuple/group_depsNoOp:^A2S/gradients/A2S/current_policy_network/add_grad/Reshape<^A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1
�
JA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependencyIdentity9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeC^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape*'
_output_shapes
:���������@
�
LA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1Identity;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1C^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*
_output_shapes
:@*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1
�
;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulMatMulJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
�
EA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul>^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1
�
MA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulF^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul
�
OA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1F^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1
�
A2S/beta1_power/initial_valueConst*
valueB
 *fff?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container 
�
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power/readIdentityA2S/beta1_power*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
A2S/beta2_power/initial_valueConst*
valueB
 *w�?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta2_power
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power/readIdentityA2S/beta2_power*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
RA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container 
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
EA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    
�
BA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
IA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
RA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
EA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
�
TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zerosConst*
_output_shapes
:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0
�
BA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@
�
IA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
�
RA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
EA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0
�
BA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
IA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zeros*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
�
RA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
EA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
�
TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
BA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
IA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1*
_output_shapes
:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
�
RA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamRA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
�
EA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@*
T0
�
TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
BA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
IA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
�
GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
RA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    
�
@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
�
GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamRA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
EA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
�
TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
BA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
IA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
S
A2S/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
S
A2S/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
U
A2S/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/w@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonOA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/b@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:@
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/w@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/b@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/w@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
use_nesterov( *
_output_shapes

:@
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/b@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
�
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
A2S/AdamNoOpR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
X
A2S/gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
A2S/gradients_1/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
A2S/gradients_1/FillFillA2S/gradients_1/ShapeA2S/gradients_1/Const*
T0*
_output_shapes
: 
~
-A2S/gradients_1/A2S/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
'A2S/gradients_1/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
%A2S/gradients_1/A2S/Mean_2_grad/ShapeShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_2_grad/TileTile'A2S/gradients_1/A2S/Mean_2_grad/Reshape%A2S/gradients_1/A2S/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
'A2S/gradients_1/A2S/Mean_2_grad/Shape_1ShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_1/A2S/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%A2S/gradients_1/A2S/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_2_grad/ProdProd'A2S/gradients_1/A2S/Mean_2_grad/Shape_1%A2S/gradients_1/A2S/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'A2S/gradients_1/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_1/A2S/Mean_2_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_2_grad/Shape_2'A2S/gradients_1/A2S/Mean_2_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_1/A2S/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
'A2S/gradients_1/A2S/Mean_2_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_2_grad/Prod_1)A2S/gradients_1/A2S/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
�
(A2S/gradients_1/A2S/Mean_2_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_2_grad/Prod'A2S/gradients_1/A2S/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
$A2S/gradients_1/A2S/Mean_2_grad/CastCast(A2S/gradients_1/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_1/A2S/Mean_2_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_2_grad/Tile$A2S/gradients_1/A2S/Mean_2_grad/Cast*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/current_value_network/add_2*
T0*
out_type0*
_output_shapes
:
}
2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1ShapeA2S/returns*
T0*
out_type0*
_output_shapes
:
�
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/current_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/NegNeg4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
;A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_depsNoOp3^A2S/gradients_1/A2S/SquaredDifference_grad/Reshape/^A2S/gradients_1/A2S/SquaredDifference_grad/Neg
�
CA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIdentity2A2S/gradients_1/A2S/SquaredDifference_grad/Reshape<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*E
_class;
97loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg*'
_output_shapes
:���������*
T0
�
:A2S/gradients_1/A2S/current_value_network/add_2_grad/ShapeShape"A2S/current_value_network/MatMul_2*
T0*
out_type0*
_output_shapes
:
�
<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
JA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8A2S/gradients_1/A2S/current_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyLA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
EA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1
�
MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
OA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1*
_output_shapes
:
�
>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:���������@*
transpose_a( 
�
@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1MatMul A2S/current_value_network/Tanh_1MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
HA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1
�
PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul*'
_output_shapes
:���������@
�
RA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@*
T0
�
>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradTanhGrad A2S/current_value_network/Tanh_1PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
:A2S/gradients_1/A2S/current_value_network/add_1_grad/ShapeShape"A2S/current_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
JA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8A2S/gradients_1/A2S/current_value_network/add_1_grad/SumSum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape*'
_output_shapes
:���������@*
T0*
Tshape0
�
:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1Sum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradLA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
EA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1
�
MA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape*'
_output_shapes
:���������@
�
OA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1*
_output_shapes
:@
�
>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(*
T0
�
@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1MatMulA2S/current_value_network/TanhMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
HA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1
�
PA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������@
�
RA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
�
<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradTanhGradA2S/current_value_network/TanhPA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
8A2S/gradients_1/A2S/current_value_network/add_grad/ShapeShape A2S/current_value_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
HA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs8A2S/gradients_1/A2S/current_value_network/add_grad/Shape:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6A2S/gradients_1/A2S/current_value_network/add_grad/SumSum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradHA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeReshape6A2S/gradients_1/A2S/current_value_network/add_grad/Sum8A2S/gradients_1/A2S/current_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1Sum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1Reshape8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
CA2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_depsNoOp;^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape=^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1
�
KA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependencyIdentity:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeD^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape*'
_output_shapes
:���������@
�
MA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1Identity<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1D^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1*
_output_shapes
:@
�
<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulMatMulKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc0/w/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
FA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul?^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1
�
NA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulG^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
PA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1G^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1*
_output_shapes

:@
�
A2S/beta1_power_1/initial_valueConst*
valueB
 *fff?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape: 
�
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
�
A2S/beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0
�
A2S/beta2_power_1
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
�
PA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/w/AdamPA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(
�
CA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w
�
RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
GA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w
�
EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
�
PA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/b/AdamPA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
CA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
�
RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container 
�
GA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
�
PA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/w/AdamPA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zeros*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
CA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam*
_output_shapes

:@@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    
�
@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container *
shape
:@@
�
GA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
�
PA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/b/AdamPA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
CA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zerosConst*
_output_shapes
:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0
�
@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@
�
GA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@
�
PA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zerosConst*
_output_shapes

:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0
�
>A2S/A2S/current_value_network/current_value_network/out/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container *
shape
:@
�
EA2S/A2S/current_value_network/current_value_network/out/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/w/AdamPA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(
�
CA2S/A2S/current_value_network/current_value_network/out/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/w/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
�
RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
GA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@
�
EA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
�
PA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
>A2S/A2S/current_value_network/current_value_network/out/b/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b
�
EA2S/A2S/current_value_network/current_value_network/out/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/b/AdamPA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
�
CA2S/A2S/current_value_network/current_value_network/out/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/b/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:
�
RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container 
�
GA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
�
EA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:*
T0
U
A2S/Adam_1/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
U
A2S/Adam_1/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
A2S/Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/w>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonPA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
use_nesterov( *
_output_shapes

:@*
use_locking( 
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/b>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
use_nesterov( *
_output_shapes
:@
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/w>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:@@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
use_nesterov( 
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/b>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/w>A2S/A2S/current_value_network/current_value_network/out/w/Adam@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/b>A2S/A2S/current_value_network/current_value_network/out/b/Adam@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( 
�
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
�

A2S/Adam_1NoOpR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
X
A2S/gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
A2S/gradients_2/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
A2S/gradients_2/FillFillA2S/gradients_2/ShapeA2S/gradients_2/Const*
T0*
_output_shapes
: 
~
-A2S/gradients_2/A2S/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
'A2S/gradients_2/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
|
%A2S/gradients_2/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
�
$A2S/gradients_2/A2S/Mean_3_grad/TileTile'A2S/gradients_2/A2S/Mean_3_grad/Reshape%A2S/gradients_2/A2S/Mean_3_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
~
'A2S/gradients_2/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_2/A2S/Mean_3_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%A2S/gradients_2/A2S/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
$A2S/gradients_2/A2S/Mean_3_grad/ProdProd'A2S/gradients_2/A2S/Mean_3_grad/Shape_1%A2S/gradients_2/A2S/Mean_3_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
'A2S/gradients_2/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_2/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_3_grad/Shape_2'A2S/gradients_2/A2S/Mean_3_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_2/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_2/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_3_grad/Prod_1)A2S/gradients_2/A2S/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
�
(A2S/gradients_2/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_3_grad/Prod'A2S/gradients_2/A2S/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
�
$A2S/gradients_2/A2S/Mean_3_grad/CastCast(A2S/gradients_2/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_2/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_3_grad/Tile$A2S/gradients_2/A2S/Mean_3_grad/Cast*'
_output_shapes
:���������*
T0
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/current_q_network/add_2*
T0*
out_type0*
_output_shapes
:

4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1ShapeA2S/returns*
_output_shapes
:*
T0*
out_type0
�
BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/current_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*
T0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/SumSum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/NegNeg6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
=A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_depsNoOp5^A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape1^A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg
�
EA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyIdentity4A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg*'
_output_shapes
:���������
�
6A2S/gradients_2/A2S/current_q_network/add_2_grad/ShapeShapeA2S/current_q_network/MatMul_2*
T0*
out_type0*
_output_shapes
:
�
8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
FA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients_2/A2S/current_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyFA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyHA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
AA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1
�
IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
KA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1*
_output_shapes
:
�
:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:���������@*
transpose_a( 
�
<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1MatMulA2S/current_q_network/Tanh_1IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
�
DA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1
�
LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul*'
_output_shapes
:���������@
�
NA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
�
:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradTanhGradA2S/current_q_network/Tanh_1LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
6A2S/gradients_2/A2S/current_q_network/add_1_grad/ShapeShapeA2S/current_q_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
FA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients_2/A2S/current_q_network/add_1_grad/SumSum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_1Sum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradHA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
AA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1
�
IA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape*'
_output_shapes
:���������@
�
KA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1*
_output_shapes
:@
�
:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1MatMulA2S/current_q_network/TanhIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
�
DA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1
�
LA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������@
�
NA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
�
8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradTanhGradA2S/current_q_network/TanhLA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
4A2S/gradients_2/A2S/current_q_network/add_grad/ShapeShapeA2S/current_q_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
DA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients_2/A2S/current_q_network/add_grad/Shape6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2A2S/gradients_2/A2S/current_q_network/add_grad/SumSum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradDA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6A2S/gradients_2/A2S/current_q_network/add_grad/ReshapeReshape2A2S/gradients_2/A2S/current_q_network/add_grad/Sum4A2S/gradients_2/A2S/current_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_1Sum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1Reshape4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_16A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
?A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_depsNoOp7^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape9^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1
�
GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients_2/A2S/current_q_network/add_grad/Reshape@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape*'
_output_shapes
:���������@
�
IA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
_output_shapes
:@*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1
�
8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulMatMulGA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
BA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul;^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1
�
JA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulC^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
LA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1C^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1
�
A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_2
VariableV2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
A2S/beta2_power_2/initial_valueConst*
valueB
 *w�?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta2_power_2
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
HA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam
VariableV2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/w/AdamHA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
;A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam*
_output_shapes

:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w
�
JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    *
dtype0
�
8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
	container *
shape
:@
�
?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
�
HA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zerosConst*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0
�
6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/b/AdamHA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
;A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zerosConst*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0
�
8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@
�
HA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zerosConst*
_output_shapes

:@@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0
�
6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/w/AdamHA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
;A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
�
JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1
VariableV2*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container *
shape
:@@*
dtype0
�
?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
�
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
�
HA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container 
�
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/b/AdamHA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
�
;A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
�
JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    
�
8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
�
?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
�
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@
�
HA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zerosConst*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    *
dtype0
�
6A2S/A2S/current_q_network/current_q_network/out/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
	container *
shape
:@
�
=A2S/A2S/current_q_network/current_q_network/out/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/w/AdamHA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
;A2S/A2S/current_q_network/current_q_network/out/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
	container 
�
?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
�
=A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
HA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    
�
6A2S/A2S/current_q_network/current_q_network/out/b/Adam
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
=A2S/A2S/current_q_network/current_q_network/out/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/b/AdamHA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
�
;A2S/A2S/current_q_network/current_q_network/out/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/b/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:
�
JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:
�
?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
=A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:*
T0
U
A2S/Adam_2/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
U
A2S/Adam_2/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
W
A2S/Adam_2/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/w6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonLA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/b6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
use_nesterov( *
_output_shapes
:@
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/w6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
use_nesterov( *
_output_shapes

:@@
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/b6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/w6A2S/A2S/current_q_network/current_q_network/out/w/Adam8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/b6A2S/A2S/current_q_network/current_q_network/out/b/Adam8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b
�
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
�

A2S/Adam_2NoOpJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1
�

A2S/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/b6A2S/best_policy_network/best_policy_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(
�
A2S/Assign_1Assign7A2S/current_policy_network/current_policy_network/fc0/w6A2S/best_policy_network/best_policy_network/fc0/w/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_2Assign7A2S/current_policy_network/current_policy_network/fc1/b6A2S/best_policy_network/best_policy_network/fc1/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_3Assign7A2S/current_policy_network/current_policy_network/fc1/w6A2S/best_policy_network/best_policy_network/fc1/w/read*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 
�
A2S/Assign_4Assign7A2S/current_policy_network/current_policy_network/out/b6A2S/best_policy_network/best_policy_network/out/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_5Assign7A2S/current_policy_network/current_policy_network/out/w6A2S/best_policy_network/best_policy_network/out/w/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_6Assign5A2S/current_value_network/current_value_network/fc0/b4A2S/best_value_network/best_value_network/fc0/b/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_7Assign5A2S/current_value_network/current_value_network/fc0/w4A2S/best_value_network/best_value_network/fc0/w/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_8Assign5A2S/current_value_network/current_value_network/fc1/b4A2S/best_value_network/best_value_network/fc1/b/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_9Assign5A2S/current_value_network/current_value_network/fc1/w4A2S/best_value_network/best_value_network/fc1/w/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_10Assign5A2S/current_value_network/current_value_network/out/b4A2S/best_value_network/best_value_network/out/b/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_11Assign5A2S/current_value_network/current_value_network/out/w4A2S/best_value_network/best_value_network/out/w/read*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
�
A2S/Assign_12Assign-A2S/current_q_network/current_q_network/fc0/b,A2S/best_q_network/best_q_network/fc0/b/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_13Assign-A2S/current_q_network/current_q_network/fc0/w,A2S/best_q_network/best_q_network/fc0/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_14Assign-A2S/current_q_network/current_q_network/fc1/b,A2S/best_q_network/best_q_network/fc1/b/read*
_output_shapes
:@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(
�
A2S/Assign_15Assign-A2S/current_q_network/current_q_network/fc1/w,A2S/best_q_network/best_q_network/fc1/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_16Assign-A2S/current_q_network/current_q_network/out/b,A2S/best_q_network/best_q_network/out/b/read*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
�
A2S/Assign_17Assign-A2S/current_q_network/current_q_network/out/w,A2S/best_q_network/best_q_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(
�
A2S/group_depsNoOp^A2S/Assign^A2S/Assign_1^A2S/Assign_2^A2S/Assign_3^A2S/Assign_4^A2S/Assign_5^A2S/Assign_6^A2S/Assign_7^A2S/Assign_8^A2S/Assign_9^A2S/Assign_10^A2S/Assign_11^A2S/Assign_12^A2S/Assign_13^A2S/Assign_14^A2S/Assign_15^A2S/Assign_16^A2S/Assign_17
�
A2S/Assign_18Assign1A2S/best_policy_network/best_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(
�
A2S/Assign_19Assign1A2S/best_policy_network/best_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
�
A2S/Assign_20Assign1A2S/best_policy_network/best_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b
�
A2S/Assign_21Assign1A2S/best_policy_network/best_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_22Assign1A2S/best_policy_network/best_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
�
A2S/Assign_23Assign1A2S/best_policy_network/best_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_24Assign/A2S/best_value_network/best_value_network/fc0/b:A2S/current_value_network/current_value_network/fc0/b/read*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
�
A2S/Assign_25Assign/A2S/best_value_network/best_value_network/fc0/w:A2S/current_value_network/current_value_network/fc0/w/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_26Assign/A2S/best_value_network/best_value_network/fc1/b:A2S/current_value_network/current_value_network/fc1/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_27Assign/A2S/best_value_network/best_value_network/fc1/w:A2S/current_value_network/current_value_network/fc1/w/read*
_output_shapes

:@@*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
validate_shape(
�
A2S/Assign_28Assign/A2S/best_value_network/best_value_network/out/b:A2S/current_value_network/current_value_network/out/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_29Assign/A2S/best_value_network/best_value_network/out/w:A2S/current_value_network/current_value_network/out/w/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_30Assign'A2S/best_q_network/best_q_network/fc0/b2A2S/current_q_network/current_q_network/fc0/b/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_31Assign'A2S/best_q_network/best_q_network/fc0/w2A2S/current_q_network/current_q_network/fc0/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_32Assign'A2S/best_q_network/best_q_network/fc1/b2A2S/current_q_network/current_q_network/fc1/b/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_33Assign'A2S/best_q_network/best_q_network/fc1/w2A2S/current_q_network/current_q_network/fc1/w/read*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0
�
A2S/Assign_34Assign'A2S/best_q_network/best_q_network/out/b2A2S/current_q_network/current_q_network/out/b/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_35Assign'A2S/best_q_network/best_q_network/out/w2A2S/current_q_network/current_q_network/out/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@
�
A2S/group_deps_1NoOp^A2S/Assign_18^A2S/Assign_19^A2S/Assign_20^A2S/Assign_21^A2S/Assign_22^A2S/Assign_23^A2S/Assign_24^A2S/Assign_25^A2S/Assign_26^A2S/Assign_27^A2S/Assign_28^A2S/Assign_29^A2S/Assign_30^A2S/Assign_31^A2S/Assign_32^A2S/Assign_33^A2S/Assign_34^A2S/Assign_35
�
A2S/Assign_36Assign1A2S/last_policy_network/last_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_37Assign1A2S/last_policy_network/last_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w
�
A2S/Assign_38Assign1A2S/last_policy_network/last_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b
�
A2S/Assign_39Assign1A2S/last_policy_network/last_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_40Assign1A2S/last_policy_network/last_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_41Assign1A2S/last_policy_network/last_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
validate_shape(*
_output_shapes

:@
x
A2S/group_deps_2NoOp^A2S/Assign_36^A2S/Assign_37^A2S/Assign_38^A2S/Assign_39^A2S/Assign_40^A2S/Assign_41
�
A2S/Merge/MergeSummaryMergeSummaryA2S/klA2S/policy_network_lossA2S/value_network_lossA2S/q_network_loss*
N*
_output_shapes
: 
n
A2S/average_reward_1/tagsConst*%
valueB BA2S/average_reward_1*
dtype0*
_output_shapes
: 
u
A2S/average_reward_1ScalarSummaryA2S/average_reward_1/tagsA2S/average_reward*
T0*
_output_shapes
: 
\
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
m

A2S/Mean_4MeanA2S/advantagesA2S/Const_4*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
A2S/average_advantage/tagsConst*&
valueB BA2S/average_advantage*
dtype0*
_output_shapes
: 
o
A2S/average_advantageScalarSummaryA2S/average_advantage/tags
A2S/Mean_4*
_output_shapes
: *
T0"���=V     �v��	�ݯbX��AJ��
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
E
Softplus
features"T
activations"T"
Ttype:
2		
W
SoftplusGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
0
Square
x"T
y"T"
Ttype:
	2	
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
�
StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.3.02v1.3.0-rc2-20-g0787eee��
s
A2S/observationsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
q
A2S/advantagesPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
V
A2S/learning_ratePlaceholder*
_output_shapes
:*
shape:*
dtype0
Y
A2S/last_mean_policyPlaceholder*
dtype0*
_output_shapes
:*
shape:
\
A2S/last_std_dev_policyPlaceholder*
dtype0*
_output_shapes
:*
shape:
n
A2S/returnsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
W
A2S/average_rewardPlaceholder*
dtype0*
_output_shapes
:*
shape:
�
XA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
seed2
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
�
RA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
7A2S/current_policy_network/current_policy_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@
�
>A2S/current_policy_network/current_policy_network/fc0/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/wRA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
<A2S/current_policy_network/current_policy_network/fc0/w/readIdentity7A2S/current_policy_network/current_policy_network/fc0/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
�
IA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    
�
7A2S/current_policy_network/current_policy_network/fc0/b
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
>A2S/current_policy_network/current_policy_network/fc0/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/bIA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
<A2S/current_policy_network/current_policy_network/fc0/b/readIdentity7A2S/current_policy_network/current_policy_network/fc0/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
�
!A2S/current_policy_network/MatMulMatMulA2S/observations<A2S/current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/current_policy_network/addAdd!A2S/current_policy_network/MatMul<A2S/current_policy_network/current_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������@
y
A2S/current_policy_network/TanhTanhA2S/current_policy_network/add*
T0*'
_output_shapes
:���������@
�
XA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB"@   @   *
dtype0
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
seed2
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
�
RA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
7A2S/current_policy_network/current_policy_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@
�
>A2S/current_policy_network/current_policy_network/fc1/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/wRA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform*
_output_shapes

:@@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(
�
<A2S/current_policy_network/current_policy_network/fc1/w/readIdentity7A2S/current_policy_network/current_policy_network/fc1/w*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@*
T0
�
IA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
7A2S/current_policy_network/current_policy_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container *
shape:@
�
>A2S/current_policy_network/current_policy_network/fc1/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/bIA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(
�
<A2S/current_policy_network/current_policy_network/fc1/b/readIdentity7A2S/current_policy_network/current_policy_network/fc1/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
�
#A2S/current_policy_network/MatMul_1MatMulA2S/current_policy_network/Tanh<A2S/current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
 A2S/current_policy_network/add_1Add#A2S/current_policy_network/MatMul_1<A2S/current_policy_network/current_policy_network/fc1/b/read*
T0*'
_output_shapes
:���������@
}
!A2S/current_policy_network/Tanh_1Tanh A2S/current_policy_network/add_1*'
_output_shapes
:���������@*
T0
�
XA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *��̽
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shape*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
seed2-*
dtype0*
_output_shapes

:@*

seed
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes
: 
�
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
RA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
7A2S/current_policy_network/current_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@
�
>A2S/current_policy_network/current_policy_network/out/w/AssignAssign7A2S/current_policy_network/current_policy_network/out/wRA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
<A2S/current_policy_network/current_policy_network/out/w/readIdentity7A2S/current_policy_network/current_policy_network/out/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
IA2S/current_policy_network/current_policy_network/out/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
7A2S/current_policy_network/current_policy_network/out/b
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
>A2S/current_policy_network/current_policy_network/out/b/AssignAssign7A2S/current_policy_network/current_policy_network/out/bIA2S/current_policy_network/current_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
<A2S/current_policy_network/current_policy_network/out/b/readIdentity7A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
�
#A2S/current_policy_network/MatMul_2MatMul!A2S/current_policy_network/Tanh_1<A2S/current_policy_network/current_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
 A2S/current_policy_network/add_2Add#A2S/current_policy_network/MatMul_2<A2S/current_policy_network/current_policy_network/out/b/read*
T0*'
_output_shapes
:���������
�
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  �?*
dtype0
�
ZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
seed2=
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
�
1A2S/best_policy_network/best_policy_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container *
shape
:@
�
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
�
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB@*    
�
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container 
�
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:@
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������@
s
A2S/best_policy_network/TanhTanhA2S/best_policy_network/add*'
_output_shapes
:���������@*
T0
�
RA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  ��
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shape*
seed2N*
dtype0*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w
�
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
�
LA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
�
1A2S/best_policy_network/best_policy_network/fc1/w
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
�
8A2S/best_policy_network/best_policy_network/fc1/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/wLA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
6A2S/best_policy_network/best_policy_network/fc1/w/readIdentity1A2S/best_policy_network/best_policy_network/fc1/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
�
CA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
1A2S/best_policy_network/best_policy_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
	container *
shape:@
�
8A2S/best_policy_network/best_policy_network/fc1/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/bCA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
6A2S/best_policy_network/best_policy_network/fc1/b/readIdentity1A2S/best_policy_network/best_policy_network/fc1/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
_output_shapes
:@
�
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/Tanh6A2S/best_policy_network/best_policy_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/best_policy_network/add_1Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/fc1/b/read*'
_output_shapes
:���������@*
T0
w
A2S/best_policy_network/Tanh_1TanhA2S/best_policy_network/add_1*
T0*'
_output_shapes
:���������@
�
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
ZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2_*
dtype0
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
�
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
�
1A2S/best_policy_network/best_policy_network/out/w
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
�
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0
�
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
�
 A2S/best_policy_network/MatMul_2MatMulA2S/best_policy_network/Tanh_16A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_26A2S/best_policy_network/best_policy_network/out/b/read*'
_output_shapes
:���������*
T0
�
RA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
seed2o*
dtype0*
_output_shapes

:@
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes
: 
�
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
�
LA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
�
1A2S/last_policy_network/last_policy_network/fc0/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
	container 
�
8A2S/last_policy_network/last_policy_network/fc0/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/wLA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
6A2S/last_policy_network/last_policy_network/fc0/w/readIdentity1A2S/last_policy_network/last_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
�
CA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
1A2S/last_policy_network/last_policy_network/fc0/b
VariableV2*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
8A2S/last_policy_network/last_policy_network/fc0/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/bCA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(
�
6A2S/last_policy_network/last_policy_network/fc0/b/readIdentity1A2S/last_policy_network/last_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
_output_shapes
:@
�
A2S/last_policy_network/MatMulMatMulA2S/observations6A2S/last_policy_network/last_policy_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/last_policy_network/addAddA2S/last_policy_network/MatMul6A2S/last_policy_network/last_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������@
s
A2S/last_policy_network/TanhTanhA2S/last_policy_network/add*
T0*'
_output_shapes
:���������@
�
RA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB"@   @   
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  ��*
dtype0
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shape*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
seed2�*
dtype0*
_output_shapes

:@@
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w
�
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/sub*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@*
T0
�
LA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
�
1A2S/last_policy_network/last_policy_network/fc1/w
VariableV2*
_output_shapes

:@@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
	container *
shape
:@@*
dtype0
�
8A2S/last_policy_network/last_policy_network/fc1/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/wLA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform*
_output_shapes

:@@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(
�
6A2S/last_policy_network/last_policy_network/fc1/w/readIdentity1A2S/last_policy_network/last_policy_network/fc1/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
�
CA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
1A2S/last_policy_network/last_policy_network/fc1/b
VariableV2*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
8A2S/last_policy_network/last_policy_network/fc1/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/bCA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b
�
6A2S/last_policy_network/last_policy_network/fc1/b/readIdentity1A2S/last_policy_network/last_policy_network/fc1/b*
_output_shapes
:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b
�
 A2S/last_policy_network/MatMul_1MatMulA2S/last_policy_network/Tanh6A2S/last_policy_network/last_policy_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/last_policy_network/add_1Add A2S/last_policy_network/MatMul_16A2S/last_policy_network/last_policy_network/fc1/b/read*'
_output_shapes
:���������@*
T0
w
A2S/last_policy_network/Tanh_1TanhA2S/last_policy_network/add_1*
T0*'
_output_shapes
:���������@
�
RA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB"@      
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *��̽*
dtype0
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *���=
�
ZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shape*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
seed2�*
dtype0*
_output_shapes

:@
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
�
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
�
LA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
�
1A2S/last_policy_network/last_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
	container *
shape
:@
�
8A2S/last_policy_network/last_policy_network/out/w/AssignAssign1A2S/last_policy_network/last_policy_network/out/wLA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
validate_shape(
�
6A2S/last_policy_network/last_policy_network/out/w/readIdentity1A2S/last_policy_network/last_policy_network/out/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
�
CA2S/last_policy_network/last_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/last_policy_network/last_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
	container *
shape:
�
8A2S/last_policy_network/last_policy_network/out/b/AssignAssign1A2S/last_policy_network/last_policy_network/out/bCA2S/last_policy_network/last_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
6A2S/last_policy_network/last_policy_network/out/b/readIdentity1A2S/last_policy_network/last_policy_network/out/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
_output_shapes
:
�
 A2S/last_policy_network/MatMul_2MatMulA2S/last_policy_network/Tanh_16A2S/last_policy_network/last_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/last_policy_network/add_2Add A2S/last_policy_network/MatMul_26A2S/last_policy_network/last_policy_network/out/b/read*'
_output_shapes
:���������*
T0
�
VA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  �?*
dtype0
�
^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
seed2�
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes
: *
T0
�
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w
�
PA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
�
5A2S/current_value_network/current_value_network/fc0/w
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
<A2S/current_value_network/current_value_network/fc0/w/AssignAssign5A2S/current_value_network/current_value_network/fc0/wPA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
:A2S/current_value_network/current_value_network/fc0/w/readIdentity5A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w
�
GA2S/current_value_network/current_value_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
5A2S/current_value_network/current_value_network/fc0/b
VariableV2*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape:@*
dtype0
�
<A2S/current_value_network/current_value_network/fc0/b/AssignAssign5A2S/current_value_network/current_value_network/fc0/bGA2S/current_value_network/current_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
:A2S/current_value_network/current_value_network/fc0/b/readIdentity5A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
 A2S/current_value_network/MatMulMatMulA2S/observations:A2S/current_value_network/current_value_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_value_network/addAdd A2S/current_value_network/MatMul:A2S/current_value_network/current_value_network/fc0/b/read*'
_output_shapes
:���������@*
T0
w
A2S/current_value_network/TanhTanhA2S/current_value_network/add*
T0*'
_output_shapes
:���������@
�
VA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  ��
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
seed2�
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
�
PA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0
�
5A2S/current_value_network/current_value_network/fc1/w
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
<A2S/current_value_network/current_value_network/fc1/w/AssignAssign5A2S/current_value_network/current_value_network/fc1/wPA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
:A2S/current_value_network/current_value_network/fc1/w/readIdentity5A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
GA2S/current_value_network/current_value_network/fc1/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
5A2S/current_value_network/current_value_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@
�
<A2S/current_value_network/current_value_network/fc1/b/AssignAssign5A2S/current_value_network/current_value_network/fc1/bGA2S/current_value_network/current_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
:A2S/current_value_network/current_value_network/fc1/b/readIdentity5A2S/current_value_network/current_value_network/fc1/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@
�
"A2S/current_value_network/MatMul_1MatMulA2S/current_value_network/Tanh:A2S/current_value_network/current_value_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_value_network/add_1Add"A2S/current_value_network/MatMul_1:A2S/current_value_network/current_value_network/fc1/b/read*'
_output_shapes
:���������@*
T0
{
 A2S/current_value_network/Tanh_1TanhA2S/current_value_network/add_1*
T0*'
_output_shapes
:���������@
�
VA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:@*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/sub*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
�
PA2S/current_value_network/current_value_network/out/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
�
5A2S/current_value_network/current_value_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container *
shape
:@
�
<A2S/current_value_network/current_value_network/out/w/AssignAssign5A2S/current_value_network/current_value_network/out/wPA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@
�
:A2S/current_value_network/current_value_network/out/w/readIdentity5A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
GA2S/current_value_network/current_value_network/out/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
5A2S/current_value_network/current_value_network/out/b
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
<A2S/current_value_network/current_value_network/out/b/AssignAssign5A2S/current_value_network/current_value_network/out/bGA2S/current_value_network/current_value_network/out/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
�
:A2S/current_value_network/current_value_network/out/b/readIdentity5A2S/current_value_network/current_value_network/out/b*
_output_shapes
:*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b
�
"A2S/current_value_network/MatMul_2MatMul A2S/current_value_network/Tanh_1:A2S/current_value_network/current_value_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/current_value_network/add_2Add"A2S/current_value_network/MatMul_2:A2S/current_value_network/current_value_network/out/b/read*
T0*'
_output_shapes
:���������
�
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"   @   *
dtype0
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  ��
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2�
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@
�
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
/A2S/best_value_network/best_value_network/fc0/b
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:@
�
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:���������@
q
A2S/best_value_network/TanhTanhA2S/best_value_network/add*'
_output_shapes
:���������@*
T0
�
PA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB"@   @   
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
XA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
seed2�
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
JA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
�
/A2S/best_value_network/best_value_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
	container *
shape
:@@
�
6A2S/best_value_network/best_value_network/fc1/w/AssignAssign/A2S/best_value_network/best_value_network/fc1/wJA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
4A2S/best_value_network/best_value_network/fc1/w/readIdentity/A2S/best_value_network/best_value_network/fc1/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
�
AA2S/best_value_network/best_value_network/fc1/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
valueB@*    
�
/A2S/best_value_network/best_value_network/fc1/b
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
6A2S/best_value_network/best_value_network/fc1/b/AssignAssign/A2S/best_value_network/best_value_network/fc1/bAA2S/best_value_network/best_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
4A2S/best_value_network/best_value_network/fc1/b/readIdentity/A2S/best_value_network/best_value_network/fc1/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
_output_shapes
:@
�
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/Tanh4A2S/best_value_network/best_value_network/fc1/w/read*
transpose_b( *
T0*'
_output_shapes
:���������@*
transpose_a( 
�
A2S/best_value_network/add_1AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/fc1/b/read*
T0*'
_output_shapes
:���������@
u
A2S/best_value_network/Tanh_1TanhA2S/best_value_network/add_1*
T0*'
_output_shapes
:���������@
�
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
XA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2�
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@
�
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@
�
/A2S/best_value_network/best_value_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:@
�
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@
�
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
AA2S/best_value_network/best_value_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    
�
/A2S/best_value_network/best_value_network/out/b
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
�
A2S/best_value_network/MatMul_2MatMulA2S/best_value_network/Tanh_14A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_24A2S/best_value_network/best_value_network/out/b/read*'
_output_shapes
:���������*
T0
h
A2S/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
j
A2S/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
j
A2S/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_sliceStridedSlice A2S/current_policy_network/add_2A2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
`
A2S/SqueezeSqueezeA2S/strided_slice*
T0*
_output_shapes
:*
squeeze_dims
 
b
A2S/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
v
A2S/ReshapeReshapeA2S/SqueezeA2S/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
l
A2S/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_slice_1StridedSliceA2S/best_policy_network/add_2A2S/strided_slice_1/stackA2S/strided_slice_1/stack_1A2S/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
d
A2S/Squeeze_1SqueezeA2S/strided_slice_1*
_output_shapes
:*
squeeze_dims
 *
T0
d
A2S/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
|
A2S/Reshape_1ReshapeA2S/Squeeze_1A2S/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"        
l
A2S/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
l
A2S/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
A2S/strided_slice_2StridedSliceA2S/last_policy_network/add_2A2S/strided_slice_2/stackA2S/strided_slice_2/stack_1A2S/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
d
A2S/Squeeze_2SqueezeA2S/strided_slice_2*
T0*
_output_shapes
:*
squeeze_dims
 
d
A2S/Reshape_2/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
|
A2S/Reshape_2ReshapeA2S/Squeeze_2A2S/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_3/stackConst*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
l
A2S/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_slice_3StridedSlice A2S/current_policy_network/add_2A2S/strided_slice_3/stackA2S/strided_slice_3/stack_1A2S/strided_slice_3/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
d
A2S/Squeeze_3SqueezeA2S/strided_slice_3*
_output_shapes
:*
squeeze_dims
 *
T0
J
A2S/SoftplusSoftplusA2S/Squeeze_3*
T0*
_output_shapes
:
N
	A2S/add/yConst*
valueB
 *��'7*
dtype0*
_output_shapes
: 
J
A2S/addAddA2S/Softplus	A2S/add/y*
T0*
_output_shapes
:
d
A2S/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
v
A2S/Reshape_3ReshapeA2S/addA2S/Reshape_3/shape*'
_output_shapes
:���������*
T0*
Tshape0
j
A2S/strided_slice_4/stackConst*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
l
A2S/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
A2S/strided_slice_4StridedSliceA2S/best_policy_network/add_2A2S/strided_slice_4/stackA2S/strided_slice_4/stack_1A2S/strided_slice_4/stack_2*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
d
A2S/Squeeze_4SqueezeA2S/strided_slice_4*
_output_shapes
:*
squeeze_dims
 *
T0
L
A2S/Softplus_1SoftplusA2S/Squeeze_4*
_output_shapes
:*
T0
P
A2S/add_1/yConst*
valueB
 *��'7*
dtype0*
_output_shapes
: 
P
	A2S/add_1AddA2S/Softplus_1A2S/add_1/y*
_output_shapes
:*
T0
d
A2S/Reshape_4/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
x
A2S/Reshape_4Reshape	A2S/add_1A2S/Reshape_4/shape*
T0*
Tshape0*'
_output_shapes
:���������
j
A2S/strided_slice_5/stackConst*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_5/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
l
A2S/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
A2S/strided_slice_5StridedSliceA2S/last_policy_network/add_2A2S/strided_slice_5/stackA2S/strided_slice_5/stack_1A2S/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
d
A2S/Squeeze_5SqueezeA2S/strided_slice_5*
_output_shapes
:*
squeeze_dims
 *
T0
L
A2S/Softplus_2SoftplusA2S/Squeeze_5*
_output_shapes
:*
T0
P
A2S/add_2/yConst*
_output_shapes
: *
valueB
 *��'7*
dtype0
P
	A2S/add_2AddA2S/Softplus_2A2S/add_2/y*
_output_shapes
:*
T0
d
A2S/Reshape_5/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
x
A2S/Reshape_5Reshape	A2S/add_2A2S/Reshape_5/shape*
T0*
Tshape0*'
_output_shapes
:���������
Y
A2S/Normal/locIdentityA2S/Reshape*
T0*'
_output_shapes
:���������
]
A2S/Normal/scaleIdentityA2S/Reshape_3*
T0*'
_output_shapes
:���������
]
A2S/Normal_1/locIdentityA2S/Reshape_1*'
_output_shapes
:���������*
T0
_
A2S/Normal_1/scaleIdentityA2S/Reshape_4*
T0*'
_output_shapes
:���������
]
A2S/Normal_2/locIdentityA2S/Reshape_2*'
_output_shapes
:���������*
T0
_
A2S/Normal_2/scaleIdentityA2S/Reshape_5*
T0*'
_output_shapes
:���������
o
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
q
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
q
,A2S/KullbackLeibler/kl_normal_normal/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
y
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal/scale*
T0*'
_output_shapes
:���������
}
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal_2/scale*
T0*'
_output_shapes
:���������
�
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*'
_output_shapes
:���������
�
(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal/locA2S/Normal_2/loc*'
_output_shapes
:���������*
T0
�
-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*'
_output_shapes
:���������*
T0
�
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*'
_output_shapes
:���������*
T0
�
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*'
_output_shapes
:���������*
T0
�
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*
T0*'
_output_shapes
:���������
�
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*'
_output_shapes
:���������*
T0
�
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*'
_output_shapes
:���������*
T0
�
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*'
_output_shapes
:���������*
T0
�
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*
T0*'
_output_shapes
:���������
Z
	A2S/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/add	A2S/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
A2S/kl/tagsConst*
valueB BA2S/kl*
dtype0*
_output_shapes
: 
O
A2S/klScalarSummaryA2S/kl/tagsA2S/Mean*
T0*
_output_shapes
: 
s
%A2S/Normal_3/batch_shape_tensor/ShapeShapeA2S/Normal/loc*
T0*
out_type0*
_output_shapes
:
w
'A2S/Normal_3/batch_shape_tensor/Shape_1ShapeA2S/Normal/scale*
T0*
out_type0*
_output_shapes
:
�
-A2S/Normal_3/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_3/batch_shape_tensor/Shape'A2S/Normal_3/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
]
A2S/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
Q
A2S/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_3/batch_shape_tensor/BroadcastArgsA2S/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
[
A2S/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
]
A2S/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*4
_output_shapes"
 :������������������*
seed2�*

seed*
T0*
dtype0
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :������������������
i
A2S/mulMulA2S/random_normalA2S/Normal/scale*
T0*+
_output_shapes
:���������
_
	A2S/add_3AddA2S/mulA2S/Normal/loc*
T0*+
_output_shapes
:���������
h
A2S/Reshape_6/shapeConst*
_output_shapes
:*!
valueB"����      *
dtype0
|
A2S/Reshape_6Reshape	A2S/add_3A2S/Reshape_6/shape*
T0*
Tshape0*+
_output_shapes
:���������
S
A2S/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*'
_output_shapes
:���������*

Tidx0*
T0*
N
�
NA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB"   @   *
dtype0
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
VA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w
�
HA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
�
-A2S/current_q_network/current_q_network/fc0/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
	container 
�
4A2S/current_q_network/current_q_network/fc0/w/AssignAssign-A2S/current_q_network/current_q_network/fc0/wHA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
2A2S/current_q_network/current_q_network/fc0/w/readIdentity-A2S/current_q_network/current_q_network/fc0/w*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@*
T0
�
?A2S/current_q_network/current_q_network/fc0/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
-A2S/current_q_network/current_q_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape:@
�
4A2S/current_q_network/current_q_network/fc0/b/AssignAssign-A2S/current_q_network/current_q_network/fc0/b?A2S/current_q_network/current_q_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(
�
2A2S/current_q_network/current_q_network/fc0/b/readIdentity-A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
A2S/current_q_network/MatMulMatMulA2S/concat_12A2S/current_q_network/current_q_network/fc0/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
A2S/current_q_network/addAddA2S/current_q_network/MatMul2A2S/current_q_network/current_q_network/fc0/b/read*
T0*'
_output_shapes
:���������@
o
A2S/current_q_network/TanhTanhA2S/current_q_network/add*'
_output_shapes
:���������@*
T0
�
NA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
VA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
seed2�
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
�
HA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@*
T0
�
-A2S/current_q_network/current_q_network/fc1/w
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container 
�
4A2S/current_q_network/current_q_network/fc1/w/AssignAssign-A2S/current_q_network/current_q_network/fc1/wHA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
2A2S/current_q_network/current_q_network/fc1/w/readIdentity-A2S/current_q_network/current_q_network/fc1/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
�
?A2S/current_q_network/current_q_network/fc1/b/Initializer/zerosConst*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0
�
-A2S/current_q_network/current_q_network/fc1/b
VariableV2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
4A2S/current_q_network/current_q_network/fc1/b/AssignAssign-A2S/current_q_network/current_q_network/fc1/b?A2S/current_q_network/current_q_network/fc1/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
2A2S/current_q_network/current_q_network/fc1/b/readIdentity-A2S/current_q_network/current_q_network/fc1/b*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@*
T0
�
A2S/current_q_network/MatMul_1MatMulA2S/current_q_network/Tanh2A2S/current_q_network/current_q_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/current_q_network/add_1AddA2S/current_q_network/MatMul_12A2S/current_q_network/current_q_network/fc1/b/read*
T0*'
_output_shapes
:���������@
s
A2S/current_q_network/Tanh_1TanhA2S/current_q_network/add_1*'
_output_shapes
:���������@*
T0
�
NA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
VA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
seed2�
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes
: 
�
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
HA2S/current_q_network/current_q_network/out/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
-A2S/current_q_network/current_q_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
	container *
shape
:@
�
4A2S/current_q_network/current_q_network/out/w/AssignAssign-A2S/current_q_network/current_q_network/out/wHA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
2A2S/current_q_network/current_q_network/out/w/readIdentity-A2S/current_q_network/current_q_network/out/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
?A2S/current_q_network/current_q_network/out/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
-A2S/current_q_network/current_q_network/out/b
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
4A2S/current_q_network/current_q_network/out/b/AssignAssign-A2S/current_q_network/current_q_network/out/b?A2S/current_q_network/current_q_network/out/b/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
2A2S/current_q_network/current_q_network/out/b/readIdentity-A2S/current_q_network/current_q_network/out/b*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:
�
A2S/current_q_network/MatMul_2MatMulA2S/current_q_network/Tanh_12A2S/current_q_network/current_q_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/current_q_network/add_2AddA2S/current_q_network/MatMul_22A2S/current_q_network/current_q_network/out/b/read*
T0*'
_output_shapes
:���������
�
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"   @   
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*
_output_shapes

:@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2�*
dtype0
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes
: *
T0
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@
�
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
'A2S/best_q_network/best_q_network/fc0/w
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@
�
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
'A2S/best_q_network/best_q_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:@
�
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
�
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:@
�
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:���������@*
transpose_a( 
�
A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*
T0*'
_output_shapes
:���������@
i
A2S/best_q_network/TanhTanhA2S/best_q_network/add*'
_output_shapes
:���������@*
T0
�
HA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB"@   @   *
dtype0
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
PA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
seed2�
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
�
BA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@*
T0
�
'A2S/best_q_network/best_q_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
	container *
shape
:@@
�
.A2S/best_q_network/best_q_network/fc1/w/AssignAssign'A2S/best_q_network/best_q_network/fc1/wBA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
,A2S/best_q_network/best_q_network/fc1/w/readIdentity'A2S/best_q_network/best_q_network/fc1/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
�
9A2S/best_q_network/best_q_network/fc1/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
'A2S/best_q_network/best_q_network/fc1/b
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
.A2S/best_q_network/best_q_network/fc1/b/AssignAssign'A2S/best_q_network/best_q_network/fc1/b9A2S/best_q_network/best_q_network/fc1/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
,A2S/best_q_network/best_q_network/fc1/b/readIdentity'A2S/best_q_network/best_q_network/fc1/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
_output_shapes
:@
�
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/Tanh,A2S/best_q_network/best_q_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
A2S/best_q_network/add_1AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/fc1/b/read*'
_output_shapes
:���������@*
T0
m
A2S/best_q_network/Tanh_1TanhA2S/best_q_network/add_1*
T0*'
_output_shapes
:���������@
�
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *���=*
dtype0
�
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@
�
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@
�
'A2S/best_q_network/best_q_network/out/w
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
9A2S/best_q_network/best_q_network/out/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
'A2S/best_q_network/best_q_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
	container *
shape:
�
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
�
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:*
T0
�
A2S/best_q_network/MatMul_2MatMulA2S/best_q_network/Tanh_1,A2S/best_q_network/best_q_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_2,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:���������
{
%A2S/Normal_4/log_prob/standardize/subSubA2S/actionsA2S/Normal/loc*
T0*'
_output_shapes
:���������
�
)A2S/Normal_4/log_prob/standardize/truedivRealDiv%A2S/Normal_4/log_prob/standardize/subA2S/Normal/scale*
T0*'
_output_shapes
:���������
�
A2S/Normal_4/log_prob/SquareSquare)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
`
A2S/Normal_4/log_prob/mul/xConst*
valueB
 *   �*
dtype0*
_output_shapes
: 
�
A2S/Normal_4/log_prob/mulMulA2S/Normal_4/log_prob/mul/xA2S/Normal_4/log_prob/Square*'
_output_shapes
:���������*
T0
d
A2S/Normal_4/log_prob/LogLogA2S/Normal/scale*'
_output_shapes
:���������*
T0
`
A2S/Normal_4/log_prob/add/xConst*
_output_shapes
: *
valueB
 *�?k?*
dtype0
�
A2S/Normal_4/log_prob/addAddA2S/Normal_4/log_prob/add/xA2S/Normal_4/log_prob/Log*'
_output_shapes
:���������*
T0
�
A2S/Normal_4/log_prob/subSubA2S/Normal_4/log_prob/mulA2S/Normal_4/log_prob/add*'
_output_shapes
:���������*
T0
[
A2S/NegNegA2S/Normal_4/log_prob/sub*'
_output_shapes
:���������*
T0
[
	A2S/mul_1MulA2S/NegA2S/advantages*
T0*'
_output_shapes
:���������
\
A2S/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
h

A2S/Mean_1Mean	A2S/mul_1A2S/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
t
A2S/policy_network_loss/tagsConst*
dtype0*
_output_shapes
: *(
valueB BA2S/policy_network_loss
s
A2S/policy_network_lossScalarSummaryA2S/policy_network_loss/tags
A2S/Mean_1*
T0*
_output_shapes
: 
�
A2S/SquaredDifferenceSquaredDifferenceA2S/current_value_network/add_2A2S/returns*'
_output_shapes
:���������*
T0
\
A2S/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
t

A2S/Mean_2MeanA2S/SquaredDifferenceA2S/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
r
A2S/value_network_loss/tagsConst*
dtype0*
_output_shapes
: *'
valueB BA2S/value_network_loss
q
A2S/value_network_lossScalarSummaryA2S/value_network_loss/tags
A2S/Mean_2*
T0*
_output_shapes
: 
�
A2S/SquaredDifference_1SquaredDifferenceA2S/current_q_network/add_2A2S/returns*'
_output_shapes
:���������*
T0
\
A2S/Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
v

A2S/Mean_3MeanA2S/SquaredDifference_1A2S/Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
A2S/q_network_loss/tagsConst*#
valueB BA2S/q_network_loss*
dtype0*
_output_shapes
: 
i
A2S/q_network_lossScalarSummaryA2S/q_network_loss/tags
A2S/Mean_3*
T0*
_output_shapes
: 
V
A2S/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
A2S/gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
A2S/gradients/FillFillA2S/gradients/ShapeA2S/gradients/Const*
_output_shapes
: *
T0
|
+A2S/gradients/A2S/Mean_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
%A2S/gradients/A2S/Mean_1_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
#A2S/gradients/A2S/Mean_1_grad/ShapeShape	A2S/mul_1*
T0*
out_type0*
_output_shapes
:
�
"A2S/gradients/A2S/Mean_1_grad/TileTile%A2S/gradients/A2S/Mean_1_grad/Reshape#A2S/gradients/A2S/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
n
%A2S/gradients/A2S/Mean_1_grad/Shape_1Shape	A2S/mul_1*
T0*
out_type0*
_output_shapes
:
h
%A2S/gradients/A2S/Mean_1_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
m
#A2S/gradients/A2S/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"A2S/gradients/A2S/Mean_1_grad/ProdProd%A2S/gradients/A2S/Mean_1_grad/Shape_1#A2S/gradients/A2S/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
o
%A2S/gradients/A2S/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients/A2S/Mean_1_grad/Prod_1Prod%A2S/gradients/A2S/Mean_1_grad/Shape_2%A2S/gradients/A2S/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
i
'A2S/gradients/A2S/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
%A2S/gradients/A2S/Mean_1_grad/MaximumMaximum$A2S/gradients/A2S/Mean_1_grad/Prod_1'A2S/gradients/A2S/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
&A2S/gradients/A2S/Mean_1_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_1_grad/Prod%A2S/gradients/A2S/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
�
"A2S/gradients/A2S/Mean_1_grad/CastCast&A2S/gradients/A2S/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
%A2S/gradients/A2S/Mean_1_grad/truedivRealDiv"A2S/gradients/A2S/Mean_1_grad/Tile"A2S/gradients/A2S/Mean_1_grad/Cast*
T0*'
_output_shapes
:���������
i
"A2S/gradients/A2S/mul_1_grad/ShapeShapeA2S/Neg*
out_type0*
_output_shapes
:*
T0
r
$A2S/gradients/A2S/mul_1_grad/Shape_1ShapeA2S/advantages*
out_type0*
_output_shapes
:*
T0
�
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_1_grad/truedivA2S/advantages*'
_output_shapes
:���������*
T0
�
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_1_grad/truediv*
T0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1
�
5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape
�
7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ShapeShapeA2S/Normal_4/log_prob/mul*
_output_shapes
:*
T0*
out_type0
�
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1ShapeA2S/Normal_4/log_prob/add*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1*
_output_shapes
:*
T0
�
6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape*'
_output_shapes
:���������
�
GA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1*'
_output_shapes
:���������
u
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1ShapeA2S/Normal_4/log_prob/Square*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_4/log_prob/Square*
T0*'
_output_shapes
:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1MulA2S/Normal_4/log_prob/mul/xEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape*
_output_shapes
: 
�
GA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1*'
_output_shapes
:���������
u
2A2S/gradients/A2S/Normal_4/log_prob/add_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
4A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape_1ShapeA2S/Normal_4/log_prob/Log*
out_type0*
_output_shapes
:*
T0
�
BA2S/gradients/A2S/Normal_4/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/add_grad/SumSumGA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1BA2S/gradients/A2S/Normal_4/log_prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients/A2S/Normal_4/log_prob/add_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/add_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Sum_1SumGA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1DA2S/gradients/A2S/Normal_4/log_prob/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_4/log_prob/add_grad/Sum_14A2S/gradients/A2S/Normal_4/log_prob/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape*
_output_shapes
: 
�
GA2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/add_grad/Reshape_1*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
3A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/x)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul*
T0*'
_output_shapes
:���������
�
7A2S/gradients/A2S/Normal_4/log_prob/Log_grad/Reciprocal
ReciprocalA2S/Normal/scaleH^A2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
0A2S/gradients/A2S/Normal_4/log_prob/Log_grad/mulMulGA2S/gradients/A2S/Normal_4/log_prob/add_grad/tuple/control_dependency_17A2S/gradients/A2S/Normal_4/log_prob/Log_grad/Reciprocal*'
_output_shapes
:���������*
T0
�
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_4/log_prob/standardize/sub*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1ShapeA2S/Normal/scale*
_output_shapes
:*
T0*
out_type0
�
RA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1A2S/Normal/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_4/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegA2S/Normal/scale*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
MA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1
�
UA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:���������
�
WA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
T0*
out_type0*
_output_shapes
:
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal/loc*
_output_shapes
:*
T0*
out_type0
�
NA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
IA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1
�
QA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:���������
�
SA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients/AddNAddN0A2S/gradients/A2S/Normal_4/log_prob/Log_grad/mulWA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependency_1*
N*'
_output_shapes
:���������*
T0*C
_class9
75loc:@A2S/gradients/A2S/Normal_4/log_prob/Log_grad/mul
v
&A2S/gradients/A2S/Reshape_3_grad/ShapeShapeA2S/add*#
_output_shapes
:���������*
T0*
out_type0
�
(A2S/gradients/A2S/Reshape_3_grad/ReshapeReshapeA2S/gradients/AddN&A2S/gradients/A2S/Reshape_3_grad/Shape*
_output_shapes
:*
T0*
Tshape0
x
$A2S/gradients/A2S/Reshape_grad/ShapeShapeA2S/Squeeze*
T0*
out_type0*#
_output_shapes
:���������
�
&A2S/gradients/A2S/Reshape_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1$A2S/gradients/A2S/Reshape_grad/Shape*
Tshape0*
_output_shapes
:*
T0
u
 A2S/gradients/A2S/add_grad/ShapeShapeA2S/Softplus*
T0*
out_type0*#
_output_shapes
:���������
e
"A2S/gradients/A2S/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
0A2S/gradients/A2S/add_grad/BroadcastGradientArgsBroadcastGradientArgs A2S/gradients/A2S/add_grad/Shape"A2S/gradients/A2S/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
A2S/gradients/A2S/add_grad/SumSum(A2S/gradients/A2S/Reshape_3_grad/Reshape0A2S/gradients/A2S/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"A2S/gradients/A2S/add_grad/ReshapeReshapeA2S/gradients/A2S/add_grad/Sum A2S/gradients/A2S/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
 A2S/gradients/A2S/add_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_3_grad/Reshape2A2S/gradients/A2S/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
$A2S/gradients/A2S/add_grad/Reshape_1Reshape A2S/gradients/A2S/add_grad/Sum_1"A2S/gradients/A2S/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+A2S/gradients/A2S/add_grad/tuple/group_depsNoOp#^A2S/gradients/A2S/add_grad/Reshape%^A2S/gradients/A2S/add_grad/Reshape_1
�
3A2S/gradients/A2S/add_grad/tuple/control_dependencyIdentity"A2S/gradients/A2S/add_grad/Reshape,^A2S/gradients/A2S/add_grad/tuple/group_deps*5
_class+
)'loc:@A2S/gradients/A2S/add_grad/Reshape*
_output_shapes
:*
T0
�
5A2S/gradients/A2S/add_grad/tuple/control_dependency_1Identity$A2S/gradients/A2S/add_grad/Reshape_1,^A2S/gradients/A2S/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@A2S/gradients/A2S/add_grad/Reshape_1*
_output_shapes
: 
u
$A2S/gradients/A2S/Squeeze_grad/ShapeShapeA2S/strided_slice*
T0*
out_type0*
_output_shapes
:
�
&A2S/gradients/A2S/Squeeze_grad/ReshapeReshape&A2S/gradients/A2S/Reshape_grad/Reshape$A2S/gradients/A2S/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
,A2S/gradients/A2S/Softplus_grad/SoftplusGradSoftplusGrad3A2S/gradients/A2S/add_grad/tuple/control_dependencyA2S/Squeeze_3*
_output_shapes
:*
T0
�
*A2S/gradients/A2S/strided_slice_grad/ShapeShape A2S/current_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
�
5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradStridedSliceGrad*A2S/gradients/A2S/strided_slice_grad/ShapeA2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2&A2S/gradients/A2S/Squeeze_grad/Reshape*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask 
y
&A2S/gradients/A2S/Squeeze_3_grad/ShapeShapeA2S/strided_slice_3*
T0*
out_type0*
_output_shapes
:
�
(A2S/gradients/A2S/Squeeze_3_grad/ReshapeReshape,A2S/gradients/A2S/Softplus_grad/SoftplusGrad&A2S/gradients/A2S/Squeeze_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
,A2S/gradients/A2S/strided_slice_3_grad/ShapeShape A2S/current_policy_network/add_2*
T0*
out_type0*
_output_shapes
:
�
7A2S/gradients/A2S/strided_slice_3_grad/StridedSliceGradStridedSliceGrad,A2S/gradients/A2S/strided_slice_3_grad/ShapeA2S/strided_slice_3/stackA2S/strided_slice_3/stack_1A2S/strided_slice_3/stack_2(A2S/gradients/A2S/Squeeze_3_grad/Reshape*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
�
A2S/gradients/AddN_1AddN5A2S/gradients/A2S/strided_slice_grad/StridedSliceGrad7A2S/gradients/A2S/strided_slice_3_grad/StridedSliceGrad*
T0*H
_class>
<:loc:@A2S/gradients/A2S/strided_slice_grad/StridedSliceGrad*
N*'
_output_shapes
:���������
�
9A2S/gradients/A2S/current_policy_network/add_2_grad/ShapeShape#A2S/current_policy_network/MatMul_2*
T0*
out_type0*
_output_shapes
:
�
;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
IA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
7A2S/gradients/A2S/current_policy_network/add_2_grad/SumSumA2S/gradients/AddN_1IA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_2_grad/Sum9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1SumA2S/gradients/AddN_1KA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
DA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1
�
LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape
�
NA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
�
=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:���������@*
transpose_a( 
�
?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1MatMul!A2S/current_policy_network/Tanh_1LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
�
GA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1
�
OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul*'
_output_shapes
:���������@
�
QA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*
_output_shapes

:@*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1
�
=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradTanhGrad!A2S/current_policy_network/Tanh_1OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
9A2S/gradients/A2S/current_policy_network/add_1_grad/ShapeShape#A2S/current_policy_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
IA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7A2S/gradients/A2S/current_policy_network/add_1_grad/SumSum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_1_grad/Sum9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1Sum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradKA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
DA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1
�
LA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape*'
_output_shapes
:���������@
�
NA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1*
_output_shapes
:@
�
=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(*
T0
�
?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/current_policy_network/TanhLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
GA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1
�
OA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul
�
QA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
�
;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradTanhGradA2S/current_policy_network/TanhOA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
7A2S/gradients/A2S/current_policy_network/add_grad/ShapeShape!A2S/current_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
GA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients/A2S/current_policy_network/add_grad/Shape9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients/A2S/current_policy_network/add_grad/SumSum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradGA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeReshape5A2S/gradients/A2S/current_policy_network/add_grad/Sum7A2S/gradients/A2S/current_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
7A2S/gradients/A2S/current_policy_network/add_grad/Sum_1Sum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1Reshape7A2S/gradients/A2S/current_policy_network/add_grad/Sum_19A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
�
BA2S/gradients/A2S/current_policy_network/add_grad/tuple/group_depsNoOp:^A2S/gradients/A2S/current_policy_network/add_grad/Reshape<^A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1
�
JA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependencyIdentity9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeC^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*L
_classB
@>loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape
�
LA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1Identity;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1C^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1*
_output_shapes
:@*
T0
�
;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulMatMulJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
EA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul>^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1
�
MA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulF^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
OA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1F^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:@
�
A2S/beta1_power/initial_valueConst*
valueB
 *fff?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta1_power
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power/readIdentityA2S/beta1_power*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
�
A2S/beta2_power/initial_valueConst*
valueB
 *w�?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta2_power
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
A2S/beta2_power/readIdentityA2S/beta2_power*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
�
RA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
�
EA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
�
TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
BA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@
�
IA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
�
RA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam
VariableV2*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@*
dtype0
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
EA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
�
TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
BA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
IA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
�
RA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zeros*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
EA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
�
TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
BA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@
�
IA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
�
RA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(
�
EA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
�
TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
BA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
�
IA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zeros*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
�
RA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w
�
GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamRA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(
�
EA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0
�
BA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@
�
IA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
�
GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
�
RA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    
�
@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:
�
GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamRA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
EA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
�
TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
BA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:
�
IA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
S
A2S/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
S
A2S/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
U
A2S/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/w@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonOA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:@*
use_locking( 
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/b@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:@
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/w@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
use_nesterov( *
_output_shapes

:@@
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/b@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/w@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w
�
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/b@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
�
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: *
T0
�
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/AdamNoOpR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
X
A2S/gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
A2S/gradients_1/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
A2S/gradients_1/FillFillA2S/gradients_1/ShapeA2S/gradients_1/Const*
_output_shapes
: *
T0
~
-A2S/gradients_1/A2S/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
'A2S/gradients_1/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
%A2S/gradients_1/A2S/Mean_2_grad/ShapeShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_2_grad/TileTile'A2S/gradients_1/A2S/Mean_2_grad/Reshape%A2S/gradients_1/A2S/Mean_2_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
|
'A2S/gradients_1/A2S/Mean_2_grad/Shape_1ShapeA2S/SquaredDifference*
out_type0*
_output_shapes
:*
T0
j
'A2S/gradients_1/A2S/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
o
%A2S/gradients_1/A2S/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_2_grad/ProdProd'A2S/gradients_1/A2S/Mean_2_grad/Shape_1%A2S/gradients_1/A2S/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
'A2S/gradients_1/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_1/A2S/Mean_2_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_2_grad/Shape_2'A2S/gradients_1/A2S/Mean_2_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
k
)A2S/gradients_1/A2S/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_1/A2S/Mean_2_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_2_grad/Prod_1)A2S/gradients_1/A2S/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
�
(A2S/gradients_1/A2S/Mean_2_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_2_grad/Prod'A2S/gradients_1/A2S/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
$A2S/gradients_1/A2S/Mean_2_grad/CastCast(A2S/gradients_1/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_1/A2S/Mean_2_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_2_grad/Tile$A2S/gradients_1/A2S/Mean_2_grad/Cast*'
_output_shapes
:���������*
T0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/current_value_network/add_2*
T0*
out_type0*
_output_shapes
:
}
2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1ShapeA2S/returns*
T0*
out_type0*
_output_shapes
:
�
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_2_grad/truediv*'
_output_shapes
:���������*
T0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/current_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_2_grad/truediv*'
_output_shapes
:���������*
T0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/NegNeg4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
;A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_depsNoOp3^A2S/gradients_1/A2S/SquaredDifference_grad/Reshape/^A2S/gradients_1/A2S/SquaredDifference_grad/Neg
�
CA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIdentity2A2S/gradients_1/A2S/SquaredDifference_grad/Reshape<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*E
_class;
97loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg
�
:A2S/gradients_1/A2S/current_value_network/add_2_grad/ShapeShape"A2S/current_value_network/MatMul_2*
T0*
out_type0*
_output_shapes
:
�
<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
JA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8A2S/gradients_1/A2S/current_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyLA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
EA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1
�
MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
OA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1*
_output_shapes
:
�
>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/out/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(*
T0
�
@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1MatMul A2S/current_value_network/Tanh_1MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
�
HA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1
�
PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul*'
_output_shapes
:���������@
�
RA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
�
>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradTanhGrad A2S/current_value_network/Tanh_1PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:���������@*
T0
�
:A2S/gradients_1/A2S/current_value_network/add_1_grad/ShapeShape"A2S/current_value_network/MatMul_1*
_output_shapes
:*
T0*
out_type0
�
<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
JA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8A2S/gradients_1/A2S/current_value_network/add_1_grad/SumSum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape*'
_output_shapes
:���������@*
T0*
Tshape0
�
:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1Sum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradLA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
EA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1
�
MA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape*'
_output_shapes
:���������@
�
OA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1
�
>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc1/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1MatMulA2S/current_value_network/TanhMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
HA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1
�
PA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������@
�
RA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
�
<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradTanhGradA2S/current_value_network/TanhPA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
8A2S/gradients_1/A2S/current_value_network/add_grad/ShapeShape A2S/current_value_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
HA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs8A2S/gradients_1/A2S/current_value_network/add_grad/Shape:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6A2S/gradients_1/A2S/current_value_network/add_grad/SumSum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradHA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeReshape6A2S/gradients_1/A2S/current_value_network/add_grad/Sum8A2S/gradients_1/A2S/current_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1Sum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1Reshape8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
Tshape0*
_output_shapes
:@*
T0
�
CA2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_depsNoOp;^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape=^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1
�
KA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependencyIdentity:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeD^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape*'
_output_shapes
:���������@
�
MA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1Identity<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1D^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1*
_output_shapes
:@
�
<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulMatMulKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
�
FA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul?^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1
�
NA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulG^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
PA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1G^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1*
_output_shapes

:@
�
A2S/beta1_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
A2S/beta1_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container 
�
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
�
A2S/beta2_power_1/initial_valueConst*
valueB
 *w�?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape: 
�
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
PA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
	container 
�
EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/w/AdamPA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
CA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w
�
RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
	container *
shape
:@
�
GA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zeros*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
�
PA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    
�
>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container 
�
EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/b/AdamPA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
CA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape:@
�
GA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(
�
EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
�
PA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/w/AdamPA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
CA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
�
RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0
�
@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container *
shape
:@@
�
GA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
�
PA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/b/AdamPA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
CA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@*
T0
�
RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
GA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@
�
PA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
>A2S/A2S/current_value_network/current_value_network/out/w/Adam
VariableV2*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container *
shape
:@*
dtype0
�
EA2S/A2S/current_value_network/current_value_network/out/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/w/AdamPA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
CA2S/A2S/current_value_network/current_value_network/out/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/w/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
�
RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container 
�
GA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
EA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
PA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
>A2S/A2S/current_value_network/current_value_network/out/b/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container 
�
EA2S/A2S/current_value_network/current_value_network/out/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/b/AdamPA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(
�
CA2S/A2S/current_value_network/current_value_network/out/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/b/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:
�
RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
GA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(
�
EA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:
U
A2S/Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
A2S/Adam_1/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
A2S/Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/w>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonPA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
use_nesterov( 
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/b>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
use_nesterov( *
_output_shapes
:@
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/w>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/b>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/w>A2S/A2S/current_value_network/current_value_network/out/w/Adam@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
use_nesterov( 
�
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/b>A2S/A2S/current_value_network/current_value_network/out/b/Adam@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
�
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( 
�

A2S/Adam_1NoOpR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
X
A2S/gradients_2/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
A2S/gradients_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
k
A2S/gradients_2/FillFillA2S/gradients_2/ShapeA2S/gradients_2/Const*
T0*
_output_shapes
: 
~
-A2S/gradients_2/A2S/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
'A2S/gradients_2/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
|
%A2S/gradients_2/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference_1*
_output_shapes
:*
T0*
out_type0
�
$A2S/gradients_2/A2S/Mean_3_grad/TileTile'A2S/gradients_2/A2S/Mean_3_grad/Reshape%A2S/gradients_2/A2S/Mean_3_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
~
'A2S/gradients_2/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_2/A2S/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
o
%A2S/gradients_2/A2S/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_2/A2S/Mean_3_grad/ProdProd'A2S/gradients_2/A2S/Mean_3_grad/Shape_1%A2S/gradients_2/A2S/Mean_3_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
'A2S/gradients_2/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_2/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_3_grad/Shape_2'A2S/gradients_2/A2S/Mean_3_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_2/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_2/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_3_grad/Prod_1)A2S/gradients_2/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
�
(A2S/gradients_2/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_3_grad/Prod'A2S/gradients_2/A2S/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
�
$A2S/gradients_2/A2S/Mean_3_grad/CastCast(A2S/gradients_2/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_2/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_3_grad/Tile$A2S/gradients_2/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/current_q_network/add_2*
T0*
out_type0*
_output_shapes
:

4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1ShapeA2S/returns*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/current_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*
T0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/SumSum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/NegNeg6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
=A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_depsNoOp5^A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape1^A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg
�
EA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyIdentity4A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg
�
6A2S/gradients_2/A2S/current_q_network/add_2_grad/ShapeShapeA2S/current_q_network/MatMul_2*
_output_shapes
:*
T0*
out_type0
�
8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
FA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients_2/A2S/current_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyFA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyHA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
AA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1
�
IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape
�
KA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1*
_output_shapes
:
�
:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/out/w/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1MatMulA2S/current_q_network/Tanh_1IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
DA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1
�
LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul*'
_output_shapes
:���������@
�
NA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@*
T0
�
:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradTanhGradA2S/current_q_network/Tanh_1LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
6A2S/gradients_2/A2S/current_q_network/add_1_grad/ShapeShapeA2S/current_q_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
FA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
4A2S/gradients_2/A2S/current_q_network/add_1_grad/SumSum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_1Sum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradHA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
�
AA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1
�
IA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape*'
_output_shapes
:���������@
�
KA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1*
_output_shapes
:@
�
:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc1/w/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(*
T0
�
<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1MatMulA2S/current_q_network/TanhIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
�
DA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1
�
LA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������@
�
NA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
�
8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradTanhGradA2S/current_q_network/TanhLA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
4A2S/gradients_2/A2S/current_q_network/add_grad/ShapeShapeA2S/current_q_network/MatMul*
_output_shapes
:*
T0*
out_type0
�
6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
�
DA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients_2/A2S/current_q_network/add_grad/Shape6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2A2S/gradients_2/A2S/current_q_network/add_grad/SumSum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradDA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
6A2S/gradients_2/A2S/current_q_network/add_grad/ReshapeReshape2A2S/gradients_2/A2S/current_q_network/add_grad/Sum4A2S/gradients_2/A2S/current_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_1Sum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1Reshape4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_16A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
�
?A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_depsNoOp7^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape9^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1
�
GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients_2/A2S/current_q_network/add_grad/Reshape@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape*'
_output_shapes
:���������@
�
IA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1*
_output_shapes
:@
�
8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulMatMulGA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
�
BA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul;^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1
�
JA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulC^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul
�
LA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1C^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1*
_output_shapes

:@
�
A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: 
�
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power_2/readIdentityA2S/beta1_power_2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: *
T0
�
A2S/beta2_power_2/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0
�
A2S/beta2_power_2
VariableV2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
HA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    
�
6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam
VariableV2*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
	container *
shape
:@*
dtype0
�
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/w/AdamHA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
;A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@*
T0
�
JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    
�
8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@*
T0
�
HA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zerosConst*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0
�
6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/b/AdamHA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
;A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape:@
�
?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
�
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
HA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
�
6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container 
�
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/w/AdamHA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
;A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam*
_output_shapes

:@@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
�
JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0
�
8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container 
�
?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1*
_output_shapes

:@@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
�
HA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    
�
6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container *
shape:@
�
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/b/AdamHA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
;A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
�
JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
�
8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container *
shape:@
�
?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
�
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@
�
HA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    
�
6A2S/A2S/current_q_network/current_q_network/out/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
	container *
shape
:@
�
=A2S/A2S/current_q_network/current_q_network/out/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/w/AdamHA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
�
;A2S/A2S/current_q_network/current_q_network/out/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
�
8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
	container *
shape
:@
�
?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
�
=A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
�
HA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
6A2S/A2S/current_q_network/current_q_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:
�
=A2S/A2S/current_q_network/current_q_network/out/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/b/AdamHA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
�
;A2S/A2S/current_q_network/current_q_network/out/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/b/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:
�
JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:
�
?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b
�
=A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:*
T0
U
A2S/Adam_2/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
A2S/Adam_2/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
A2S/Adam_2/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/w6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonLA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
use_nesterov( *
_output_shapes

:@
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/b6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
use_nesterov( *
_output_shapes
:@
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/w6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:@@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
use_nesterov( 
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/b6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
use_nesterov( *
_output_shapes
:@
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/w6A2S/A2S/current_q_network/current_q_network/out/w/Adam8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
use_nesterov( *
_output_shapes

:@
�
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/b6A2S/A2S/current_q_network/current_q_network/out/b/Adam8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
�

A2S/Adam_2NoOpJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1
�

A2S/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/b6A2S/best_policy_network/best_policy_network/fc0/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_1Assign7A2S/current_policy_network/current_policy_network/fc0/w6A2S/best_policy_network/best_policy_network/fc0/w/read*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
�
A2S/Assign_2Assign7A2S/current_policy_network/current_policy_network/fc1/b6A2S/best_policy_network/best_policy_network/fc1/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_3Assign7A2S/current_policy_network/current_policy_network/fc1/w6A2S/best_policy_network/best_policy_network/fc1/w/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_4Assign7A2S/current_policy_network/current_policy_network/out/b6A2S/best_policy_network/best_policy_network/out/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_5Assign7A2S/current_policy_network/current_policy_network/out/w6A2S/best_policy_network/best_policy_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(
�
A2S/Assign_6Assign5A2S/current_value_network/current_value_network/fc0/b4A2S/best_value_network/best_value_network/fc0/b/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_7Assign5A2S/current_value_network/current_value_network/fc0/w4A2S/best_value_network/best_value_network/fc0/w/read*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(
�
A2S/Assign_8Assign5A2S/current_value_network/current_value_network/fc1/b4A2S/best_value_network/best_value_network/fc1/b/read*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking( 
�
A2S/Assign_9Assign5A2S/current_value_network/current_value_network/fc1/w4A2S/best_value_network/best_value_network/fc1/w/read*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
�
A2S/Assign_10Assign5A2S/current_value_network/current_value_network/out/b4A2S/best_value_network/best_value_network/out/b/read*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
�
A2S/Assign_11Assign5A2S/current_value_network/current_value_network/out/w4A2S/best_value_network/best_value_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
�
A2S/Assign_12Assign-A2S/current_q_network/current_q_network/fc0/b,A2S/best_q_network/best_q_network/fc0/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
�
A2S/Assign_13Assign-A2S/current_q_network/current_q_network/fc0/w,A2S/best_q_network/best_q_network/fc0/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_14Assign-A2S/current_q_network/current_q_network/fc1/b,A2S/best_q_network/best_q_network/fc1/b/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_15Assign-A2S/current_q_network/current_q_network/fc1/w,A2S/best_q_network/best_q_network/fc1/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_16Assign-A2S/current_q_network/current_q_network/out/b,A2S/best_q_network/best_q_network/out/b/read*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
�
A2S/Assign_17Assign-A2S/current_q_network/current_q_network/out/w,A2S/best_q_network/best_q_network/out/w/read*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( 
�
A2S/group_depsNoOp^A2S/Assign^A2S/Assign_1^A2S/Assign_2^A2S/Assign_3^A2S/Assign_4^A2S/Assign_5^A2S/Assign_6^A2S/Assign_7^A2S/Assign_8^A2S/Assign_9^A2S/Assign_10^A2S/Assign_11^A2S/Assign_12^A2S/Assign_13^A2S/Assign_14^A2S/Assign_15^A2S/Assign_16^A2S/Assign_17
�
A2S/Assign_18Assign1A2S/best_policy_network/best_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_19Assign1A2S/best_policy_network/best_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_20Assign1A2S/best_policy_network/best_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking( 
�
A2S/Assign_21Assign1A2S/best_policy_network/best_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_22Assign1A2S/best_policy_network/best_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_23Assign1A2S/best_policy_network/best_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
A2S/Assign_24Assign/A2S/best_value_network/best_value_network/fc0/b:A2S/current_value_network/current_value_network/fc0/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_25Assign/A2S/best_value_network/best_value_network/fc0/w:A2S/current_value_network/current_value_network/fc0/w/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_26Assign/A2S/best_value_network/best_value_network/fc1/b:A2S/current_value_network/current_value_network/fc1/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_27Assign/A2S/best_value_network/best_value_network/fc1/w:A2S/current_value_network/current_value_network/fc1/w/read*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w
�
A2S/Assign_28Assign/A2S/best_value_network/best_value_network/out/b:A2S/current_value_network/current_value_network/out/b/read*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
�
A2S/Assign_29Assign/A2S/best_value_network/best_value_network/out/w:A2S/current_value_network/current_value_network/out/w/read*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
�
A2S/Assign_30Assign'A2S/best_q_network/best_q_network/fc0/b2A2S/current_q_network/current_q_network/fc0/b/read*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
�
A2S/Assign_31Assign'A2S/best_q_network/best_q_network/fc0/w2A2S/current_q_network/current_q_network/fc0/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
�
A2S/Assign_32Assign'A2S/best_q_network/best_q_network/fc1/b2A2S/current_q_network/current_q_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b
�
A2S/Assign_33Assign'A2S/best_q_network/best_q_network/fc1/w2A2S/current_q_network/current_q_network/fc1/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_34Assign'A2S/best_q_network/best_q_network/out/b2A2S/current_q_network/current_q_network/out/b/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
�
A2S/Assign_35Assign'A2S/best_q_network/best_q_network/out/w2A2S/current_q_network/current_q_network/out/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@
�
A2S/group_deps_1NoOp^A2S/Assign_18^A2S/Assign_19^A2S/Assign_20^A2S/Assign_21^A2S/Assign_22^A2S/Assign_23^A2S/Assign_24^A2S/Assign_25^A2S/Assign_26^A2S/Assign_27^A2S/Assign_28^A2S/Assign_29^A2S/Assign_30^A2S/Assign_31^A2S/Assign_32^A2S/Assign_33^A2S/Assign_34^A2S/Assign_35
�
A2S/Assign_36Assign1A2S/last_policy_network/last_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( 
�
A2S/Assign_37Assign1A2S/last_policy_network/last_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
�
A2S/Assign_38Assign1A2S/last_policy_network/last_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
�
A2S/Assign_39Assign1A2S/last_policy_network/last_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
�
A2S/Assign_40Assign1A2S/last_policy_network/last_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*
_output_shapes
:*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(
�
A2S/Assign_41Assign1A2S/last_policy_network/last_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
x
A2S/group_deps_2NoOp^A2S/Assign_36^A2S/Assign_37^A2S/Assign_38^A2S/Assign_39^A2S/Assign_40^A2S/Assign_41
�
A2S/Merge/MergeSummaryMergeSummaryA2S/klA2S/policy_network_lossA2S/value_network_lossA2S/q_network_loss*
N*
_output_shapes
: 
n
A2S/average_reward_1/tagsConst*%
valueB BA2S/average_reward_1*
dtype0*
_output_shapes
: 
u
A2S/average_reward_1ScalarSummaryA2S/average_reward_1/tagsA2S/average_reward*
_output_shapes
: *
T0
\
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
m

A2S/Mean_4MeanA2S/advantagesA2S/Const_4*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
A2S/average_advantage/tagsConst*&
valueB BA2S/average_advantage*
dtype0*
_output_shapes
: 
o
A2S/average_advantageScalarSummaryA2S/average_advantage/tags
A2S/Mean_4*
T0*
_output_shapes
: ""�
	summaries�
�
A2S/kl:0
A2S/policy_network_loss:0
A2S/value_network_loss:0
A2S/q_network_loss:0
A2S/average_reward_1:0
A2S/average_advantage:0"�7
trainable_variables�7�7
�
9A2S/current_policy_network/current_policy_network/fc0/w:0>A2S/current_policy_network/current_policy_network/fc0/w/Assign>A2S/current_policy_network/current_policy_network/fc0/w/read:0
�
9A2S/current_policy_network/current_policy_network/fc0/b:0>A2S/current_policy_network/current_policy_network/fc0/b/Assign>A2S/current_policy_network/current_policy_network/fc0/b/read:0
�
9A2S/current_policy_network/current_policy_network/fc1/w:0>A2S/current_policy_network/current_policy_network/fc1/w/Assign>A2S/current_policy_network/current_policy_network/fc1/w/read:0
�
9A2S/current_policy_network/current_policy_network/fc1/b:0>A2S/current_policy_network/current_policy_network/fc1/b/Assign>A2S/current_policy_network/current_policy_network/fc1/b/read:0
�
9A2S/current_policy_network/current_policy_network/out/w:0>A2S/current_policy_network/current_policy_network/out/w/Assign>A2S/current_policy_network/current_policy_network/out/w/read:0
�
9A2S/current_policy_network/current_policy_network/out/b:0>A2S/current_policy_network/current_policy_network/out/b/Assign>A2S/current_policy_network/current_policy_network/out/b/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0
�
3A2S/best_policy_network/best_policy_network/fc1/w:08A2S/best_policy_network/best_policy_network/fc1/w/Assign8A2S/best_policy_network/best_policy_network/fc1/w/read:0
�
3A2S/best_policy_network/best_policy_network/fc1/b:08A2S/best_policy_network/best_policy_network/fc1/b/Assign8A2S/best_policy_network/best_policy_network/fc1/b/read:0
�
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
�
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
�
3A2S/last_policy_network/last_policy_network/fc0/w:08A2S/last_policy_network/last_policy_network/fc0/w/Assign8A2S/last_policy_network/last_policy_network/fc0/w/read:0
�
3A2S/last_policy_network/last_policy_network/fc0/b:08A2S/last_policy_network/last_policy_network/fc0/b/Assign8A2S/last_policy_network/last_policy_network/fc0/b/read:0
�
3A2S/last_policy_network/last_policy_network/fc1/w:08A2S/last_policy_network/last_policy_network/fc1/w/Assign8A2S/last_policy_network/last_policy_network/fc1/w/read:0
�
3A2S/last_policy_network/last_policy_network/fc1/b:08A2S/last_policy_network/last_policy_network/fc1/b/Assign8A2S/last_policy_network/last_policy_network/fc1/b/read:0
�
3A2S/last_policy_network/last_policy_network/out/w:08A2S/last_policy_network/last_policy_network/out/w/Assign8A2S/last_policy_network/last_policy_network/out/w/read:0
�
3A2S/last_policy_network/last_policy_network/out/b:08A2S/last_policy_network/last_policy_network/out/b/Assign8A2S/last_policy_network/last_policy_network/out/b/read:0
�
7A2S/current_value_network/current_value_network/fc0/w:0<A2S/current_value_network/current_value_network/fc0/w/Assign<A2S/current_value_network/current_value_network/fc0/w/read:0
�
7A2S/current_value_network/current_value_network/fc0/b:0<A2S/current_value_network/current_value_network/fc0/b/Assign<A2S/current_value_network/current_value_network/fc0/b/read:0
�
7A2S/current_value_network/current_value_network/fc1/w:0<A2S/current_value_network/current_value_network/fc1/w/Assign<A2S/current_value_network/current_value_network/fc1/w/read:0
�
7A2S/current_value_network/current_value_network/fc1/b:0<A2S/current_value_network/current_value_network/fc1/b/Assign<A2S/current_value_network/current_value_network/fc1/b/read:0
�
7A2S/current_value_network/current_value_network/out/w:0<A2S/current_value_network/current_value_network/out/w/Assign<A2S/current_value_network/current_value_network/out/w/read:0
�
7A2S/current_value_network/current_value_network/out/b:0<A2S/current_value_network/current_value_network/out/b/Assign<A2S/current_value_network/current_value_network/out/b/read:0
�
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
�
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0
�
1A2S/best_value_network/best_value_network/fc1/w:06A2S/best_value_network/best_value_network/fc1/w/Assign6A2S/best_value_network/best_value_network/fc1/w/read:0
�
1A2S/best_value_network/best_value_network/fc1/b:06A2S/best_value_network/best_value_network/fc1/b/Assign6A2S/best_value_network/best_value_network/fc1/b/read:0
�
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
�
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0
�
/A2S/current_q_network/current_q_network/fc0/w:04A2S/current_q_network/current_q_network/fc0/w/Assign4A2S/current_q_network/current_q_network/fc0/w/read:0
�
/A2S/current_q_network/current_q_network/fc0/b:04A2S/current_q_network/current_q_network/fc0/b/Assign4A2S/current_q_network/current_q_network/fc0/b/read:0
�
/A2S/current_q_network/current_q_network/fc1/w:04A2S/current_q_network/current_q_network/fc1/w/Assign4A2S/current_q_network/current_q_network/fc1/w/read:0
�
/A2S/current_q_network/current_q_network/fc1/b:04A2S/current_q_network/current_q_network/fc1/b/Assign4A2S/current_q_network/current_q_network/fc1/b/read:0
�
/A2S/current_q_network/current_q_network/out/w:04A2S/current_q_network/current_q_network/out/w/Assign4A2S/current_q_network/current_q_network/out/w/read:0
�
/A2S/current_q_network/current_q_network/out/b:04A2S/current_q_network/current_q_network/out/b/Assign4A2S/current_q_network/current_q_network/out/b/read:0
�
)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0
�
)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0
�
)A2S/best_q_network/best_q_network/fc1/w:0.A2S/best_q_network/best_q_network/fc1/w/Assign.A2S/best_q_network/best_q_network/fc1/w/read:0
�
)A2S/best_q_network/best_q_network/fc1/b:0.A2S/best_q_network/best_q_network/fc1/b/Assign.A2S/best_q_network/best_q_network/fc1/b/read:0
�
)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0
�
)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0"0
train_op$
"
A2S/Adam

A2S/Adam_1

A2S/Adam_2"�u
	variables�u�u
�
9A2S/current_policy_network/current_policy_network/fc0/w:0>A2S/current_policy_network/current_policy_network/fc0/w/Assign>A2S/current_policy_network/current_policy_network/fc0/w/read:0
�
9A2S/current_policy_network/current_policy_network/fc0/b:0>A2S/current_policy_network/current_policy_network/fc0/b/Assign>A2S/current_policy_network/current_policy_network/fc0/b/read:0
�
9A2S/current_policy_network/current_policy_network/fc1/w:0>A2S/current_policy_network/current_policy_network/fc1/w/Assign>A2S/current_policy_network/current_policy_network/fc1/w/read:0
�
9A2S/current_policy_network/current_policy_network/fc1/b:0>A2S/current_policy_network/current_policy_network/fc1/b/Assign>A2S/current_policy_network/current_policy_network/fc1/b/read:0
�
9A2S/current_policy_network/current_policy_network/out/w:0>A2S/current_policy_network/current_policy_network/out/w/Assign>A2S/current_policy_network/current_policy_network/out/w/read:0
�
9A2S/current_policy_network/current_policy_network/out/b:0>A2S/current_policy_network/current_policy_network/out/b/Assign>A2S/current_policy_network/current_policy_network/out/b/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0
�
3A2S/best_policy_network/best_policy_network/fc1/w:08A2S/best_policy_network/best_policy_network/fc1/w/Assign8A2S/best_policy_network/best_policy_network/fc1/w/read:0
�
3A2S/best_policy_network/best_policy_network/fc1/b:08A2S/best_policy_network/best_policy_network/fc1/b/Assign8A2S/best_policy_network/best_policy_network/fc1/b/read:0
�
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
�
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
�
3A2S/last_policy_network/last_policy_network/fc0/w:08A2S/last_policy_network/last_policy_network/fc0/w/Assign8A2S/last_policy_network/last_policy_network/fc0/w/read:0
�
3A2S/last_policy_network/last_policy_network/fc0/b:08A2S/last_policy_network/last_policy_network/fc0/b/Assign8A2S/last_policy_network/last_policy_network/fc0/b/read:0
�
3A2S/last_policy_network/last_policy_network/fc1/w:08A2S/last_policy_network/last_policy_network/fc1/w/Assign8A2S/last_policy_network/last_policy_network/fc1/w/read:0
�
3A2S/last_policy_network/last_policy_network/fc1/b:08A2S/last_policy_network/last_policy_network/fc1/b/Assign8A2S/last_policy_network/last_policy_network/fc1/b/read:0
�
3A2S/last_policy_network/last_policy_network/out/w:08A2S/last_policy_network/last_policy_network/out/w/Assign8A2S/last_policy_network/last_policy_network/out/w/read:0
�
3A2S/last_policy_network/last_policy_network/out/b:08A2S/last_policy_network/last_policy_network/out/b/Assign8A2S/last_policy_network/last_policy_network/out/b/read:0
�
7A2S/current_value_network/current_value_network/fc0/w:0<A2S/current_value_network/current_value_network/fc0/w/Assign<A2S/current_value_network/current_value_network/fc0/w/read:0
�
7A2S/current_value_network/current_value_network/fc0/b:0<A2S/current_value_network/current_value_network/fc0/b/Assign<A2S/current_value_network/current_value_network/fc0/b/read:0
�
7A2S/current_value_network/current_value_network/fc1/w:0<A2S/current_value_network/current_value_network/fc1/w/Assign<A2S/current_value_network/current_value_network/fc1/w/read:0
�
7A2S/current_value_network/current_value_network/fc1/b:0<A2S/current_value_network/current_value_network/fc1/b/Assign<A2S/current_value_network/current_value_network/fc1/b/read:0
�
7A2S/current_value_network/current_value_network/out/w:0<A2S/current_value_network/current_value_network/out/w/Assign<A2S/current_value_network/current_value_network/out/w/read:0
�
7A2S/current_value_network/current_value_network/out/b:0<A2S/current_value_network/current_value_network/out/b/Assign<A2S/current_value_network/current_value_network/out/b/read:0
�
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
�
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0
�
1A2S/best_value_network/best_value_network/fc1/w:06A2S/best_value_network/best_value_network/fc1/w/Assign6A2S/best_value_network/best_value_network/fc1/w/read:0
�
1A2S/best_value_network/best_value_network/fc1/b:06A2S/best_value_network/best_value_network/fc1/b/Assign6A2S/best_value_network/best_value_network/fc1/b/read:0
�
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
�
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0
�
/A2S/current_q_network/current_q_network/fc0/w:04A2S/current_q_network/current_q_network/fc0/w/Assign4A2S/current_q_network/current_q_network/fc0/w/read:0
�
/A2S/current_q_network/current_q_network/fc0/b:04A2S/current_q_network/current_q_network/fc0/b/Assign4A2S/current_q_network/current_q_network/fc0/b/read:0
�
/A2S/current_q_network/current_q_network/fc1/w:04A2S/current_q_network/current_q_network/fc1/w/Assign4A2S/current_q_network/current_q_network/fc1/w/read:0
�
/A2S/current_q_network/current_q_network/fc1/b:04A2S/current_q_network/current_q_network/fc1/b/Assign4A2S/current_q_network/current_q_network/fc1/b/read:0
�
/A2S/current_q_network/current_q_network/out/w:04A2S/current_q_network/current_q_network/out/w/Assign4A2S/current_q_network/current_q_network/out/w/read:0
�
/A2S/current_q_network/current_q_network/out/b:04A2S/current_q_network/current_q_network/out/b/Assign4A2S/current_q_network/current_q_network/out/b/read:0
�
)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0
�
)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0
�
)A2S/best_q_network/best_q_network/fc1/w:0.A2S/best_q_network/best_q_network/fc1/w/Assign.A2S/best_q_network/best_q_network/fc1/w/read:0
�
)A2S/best_q_network/best_q_network/fc1/b:0.A2S/best_q_network/best_q_network/fc1/b/Assign.A2S/best_q_network/best_q_network/fc1/b/read:0
�
)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0
�
)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0
C
A2S/beta1_power:0A2S/beta1_power/AssignA2S/beta1_power/read:0
C
A2S/beta2_power:0A2S/beta2_power/AssignA2S/beta2_power/read:0
�
BA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/read:0
�
DA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/read:0
�
BA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/read:0
�
DA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/read:0
�
BA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/read:0
�
DA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/read:0
�
BA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/read:0
�
DA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/read:0
�
BA2S/A2S/current_policy_network/current_policy_network/out/w/Adam:0GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/read:0
�
DA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/read:0
�
BA2S/A2S/current_policy_network/current_policy_network/out/b/Adam:0GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/read:0
�
DA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/read:0
I
A2S/beta1_power_1:0A2S/beta1_power_1/AssignA2S/beta1_power_1/read:0
I
A2S/beta2_power_1:0A2S/beta2_power_1/AssignA2S/beta2_power_1/read:0
�
@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam:0EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/read:0
�
BA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/read:0
�
@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam:0EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/read:0
�
BA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/read:0
�
@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam:0EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/read:0
�
BA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/read:0
�
@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam:0EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/read:0
�
BA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/read:0
�
@A2S/A2S/current_value_network/current_value_network/out/w/Adam:0EA2S/A2S/current_value_network/current_value_network/out/w/Adam/AssignEA2S/A2S/current_value_network/current_value_network/out/w/Adam/read:0
�
BA2S/A2S/current_value_network/current_value_network/out/w/Adam_1:0GA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/read:0
�
@A2S/A2S/current_value_network/current_value_network/out/b/Adam:0EA2S/A2S/current_value_network/current_value_network/out/b/Adam/AssignEA2S/A2S/current_value_network/current_value_network/out/b/Adam/read:0
�
BA2S/A2S/current_value_network/current_value_network/out/b/Adam_1:0GA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/read:0
I
A2S/beta1_power_2:0A2S/beta1_power_2/AssignA2S/beta1_power_2/read:0
I
A2S/beta2_power_2:0A2S/beta2_power_2/AssignA2S/beta2_power_2/read:0
�
8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam:0=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/read:0
�
:A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/read:0
�
8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam:0=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/read:0
�
:A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/read:0
�
8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam:0=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/read:0
�
:A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/read:0
�
8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam:0=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/read:0
�
:A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/read:0
�
8A2S/A2S/current_q_network/current_q_network/out/w/Adam:0=A2S/A2S/current_q_network/current_q_network/out/w/Adam/Assign=A2S/A2S/current_q_network/current_q_network/out/w/Adam/read:0
�
:A2S/A2S/current_q_network/current_q_network/out/w/Adam_1:0?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/read:0
�
8A2S/A2S/current_q_network/current_q_network/out/b/Adam:0=A2S/A2S/current_q_network/current_q_network/out/b/Adam/Assign=A2S/A2S/current_q_network/current_q_network/out/b/Adam/read:0
�
:A2S/A2S/current_q_network/current_q_network/out/b/Adam_1:0?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/read:0��J�*       ����	�U�bX��A	*

A2S/average_reward_1T�AV	��*       ����	b�bX��A*

A2S/average_reward_1/��A28�*       ����	��bX��A*

A2S/average_reward_1)0�AR�(�*       ����	ё�bX��A$*

A2S/average_reward_1��A��7*       ����		��bX��A)*

A2S/average_reward_1!4A���*       ����	���bX��A9*

A2S/average_reward_1ί�A�F(*       ����	���bX��AI*

A2S/average_reward_1(�A�scC*       ����	%�cX��Ab*

A2S/average_reward_14��@�E�~*       ����	�mcX��Am*

A2S/average_reward_1$ԎA�J�*       ����	�5
cX��Ay*

A2S/average_reward_1��qA� �+       ��K	�XcX��A�*

A2S/average_reward_1# �A.��+       ��K	d�cX��A�*

A2S/average_reward_1�ĊA+�ڕ+       ��K	#�cX��A�*

A2S/average_reward_1��A��+       ��K	PacX��A�*

A2S/average_reward_1Z�uA'�+       ��K	J�cX��A�*

A2S/average_reward_1Z�A*�J+       ��K	�zcX��A�*

A2S/average_reward_1��A���m+       ��K	�cX��A�*

A2S/average_reward_1R=�A�<��+       ��K	h� cX��A�*

A2S/average_reward_1n?�Aex�+       ��K	ج"cX��A�*

A2S/average_reward_1��iAgT�+       ��K	� %cX��A�*

A2S/average_reward_1���A�C*�+       ��K	h�'cX��A�*

A2S/average_reward_1��{AaGV+       ��K	O�*cX��A�*

A2S/average_reward_1��A �+       ��K	h/cX��A�*

A2S/average_reward_1	��A=!<=+       ��K	��3cX��A�*

A2S/average_reward_1�ѐA�n�"+       ��K	�w;cX��A�*

A2S/average_reward_13V�A ���+       ��K	A@cX��A�*

A2S/average_reward_1��ACɩ�+       ��K	�DcX��A�*

A2S/average_reward_1�яA��٪+       ��K	s�HcX��A�*

A2S/average_reward_1�-A0�|+       ��K	��JcX��A�*

A2S/average_reward_1t�`AZ�-+       ��K	>�McX��A�*

A2S/average_reward_1D3�A����+       ��K	3WcX��A�*

A2S/average_reward_1�}B�^ܳ+       ��K	YcX��A�*

A2S/average_reward_1þ�A5�+       ��K	n�ZcX��A�*

A2S/average_reward_1]_-Aa��+       ��K	�4acX��A�*

A2S/average_reward_1ަA�I��+       ��K	f�dcX��A�*

A2S/average_reward_1���A��np+       ��K	S�gcX��A�*

A2S/average_reward_1��tA��L�+       ��K	��mcX��A�*

A2S/average_reward_1���AM�_+       ��K	U�pcX��A�*

A2S/average_reward_1^:�Am(��+       ��K	c�scX��A�*

A2S/average_reward_1�EA�c�+       ��K	�vcX��A�*

A2S/average_reward_1u�$A�Oog+       ��K	�+ycX��A�*

A2S/average_reward_1闌A��+       ��K	�||cX��A�*

A2S/average_reward_1��AYc��+       ��K	;�cX��A�*

A2S/average_reward_1�\�A����+       ��K	��cX��A�*

A2S/average_reward_1hu�A.�[w       洦�	WQlX��A�*i

A2S/kl��?

A2S/policy_network_loss��|�

A2S/value_network_loss�t�A

A2S/q_network_loss���A�=��+       ��K	:�VlX��A�*

A2S/average_reward_1V��A׬�+       ��K	�/]lX��A�*

A2S/average_reward_1�òA���Y+       ��K	��qlX��A�*

A2S/average_reward_1ȱ��u&�+       ��K	mu�lX��A�*

A2S/average_reward_1���AA�$+       ��K	���lX��A�*

A2S/average_reward_1�C�A�=�+       ��K	2�lX��A�*

A2S/average_reward_1)��A.�$�+       ��K	j��lX��A�*

A2S/average_reward_1�"�A�� R+       ��K	^�lX��A�*

A2S/average_reward_1���AD��+       ��K	l�lX��A�*

A2S/average_reward_1{"�A�eMt+       ��K	���lX��A�*

A2S/average_reward_1Mk�Atk�/+       ��K	���lX��A�*

A2S/average_reward_1��A�5�+       ��K	�[�lX��A�*

A2S/average_reward_1;��A����+       ��K	�n�lX��A�*

A2S/average_reward_1�ΦA �%g+       ��K	n�lX��A�*

A2S/average_reward_1t��An���+       ��K	`fmX��A�*

A2S/average_reward_1	�B}Q�:+       ��K	ܵmX��A�*

A2S/average_reward_1���A����+       ��K	c�mX��A�*

A2S/average_reward_1`�jA���+       ��K	G�&mX��A�*

A2S/average_reward_1�:�AO�0w       洦�	�P�uX��A�*i

A2S/kl��:

A2S/policy_network_loss��

A2S/value_network_loss���B

A2S/q_network_loss�ՑB�,�+       ��K	B��uX��A�*

A2S/average_reward_1c�Aݺ(+       ��K	��uX��A�*

A2S/average_reward_1��3ApS+       ��K	�~�uX��A�*

A2S/average_reward_1�7�AB�ly+       ��K	l�uX��A�	*

A2S/average_reward_1�އAu���+       ��K	�uX��A�	*

A2S/average_reward_1��gA�	r.+       ��K	���uX��A�	*

A2S/average_reward_1��A���+       ��K	+��uX��A�	*

A2S/average_reward_1'S�@�E\.+       ��K	���uX��A�	*

A2S/average_reward_1�c�A�fn+       ��K	��uX��A�
*

A2S/average_reward_1�bAi:!+       ��K	�vX��A�
*

A2S/average_reward_1��@P"ğ+       ��K	r&vX��A�
*

A2S/average_reward_1��qA}V9+       ��K	�N#vX��A�
*

A2S/average_reward_1�j�AJI�+       ��K	j�/vX��A�*

A2S/average_reward_1kNB�+       ��K	!5vX��A�*

A2S/average_reward_1_�`A���+       ��K	/�7vX��A�*

A2S/average_reward_1t��A���+       ��K	J�CvX��A�*

A2S/average_reward_1k9�A�M+       ��K	bNvX��A�*

A2S/average_reward_12>�A���[+       ��K	�UvX��A�*

A2S/average_reward_1��A�?��w       洦�	��~X��A�*i

A2S/kl
`�:

A2S/policy_network_loss�b��

A2S/value_network_loss��B

A2S/q_network_loss�|B$�-+       ��K	�7�~X��A�*

A2S/average_reward_1-Z�A��&(+       ��K	C�~X��A�*

A2S/average_reward_1��A۩#�+       ��K	Uؽ~X��A�*

A2S/average_reward_1�o�A:f�+       ��K	߰�~X��A�*

A2S/average_reward_1j8uA�=�+       ��K	��~X��A�*

A2S/average_reward_1��A�L�s+       ��K	\��~X��A�*

A2S/average_reward_1���AZ7��+       ��K	@�~X��A�*

A2S/average_reward_13��Aw��y+       ��K	(�~X��A�*

A2S/average_reward_1ȉA��j�+       ��K	Ҵ�~X��A�*

A2S/average_reward_1�]�Awx+       ��K	��~X��A�*

A2S/average_reward_1+�AJ� �+       ��K	[�~X��A�*

A2S/average_reward_1m��@@Lb+       ��K	oL�~X��A�*

A2S/average_reward_1��"A����+       ��K	ڔ X��A�*

A2S/average_reward_1�p�A̐��+       ��K	 �X��A�*

A2S/average_reward_1���A�+h�+       ��K	l�X��A�*

A2S/average_reward_1(T�A��m+       ��K	��X��A�*

A2S/average_reward_1@��A%&se+       ��K	�X��A�*

A2S/average_reward_1O��A3Ĵ+       ��K	�#X��A�*

A2S/average_reward_1}��A�PY�+       ��K	��X��A�*

A2S/average_reward_1���A�lMq+       ��K	.X��A�*

A2S/average_reward_1U`�A؊K�+       ��K	�X��A�*

A2S/average_reward_1��A�A+       ��K	� "X��A�*

A2S/average_reward_1���AR�	�+       ��K	�%X��A�*

A2S/average_reward_1�\�A]��)+       ��K	J(X��A�*

A2S/average_reward_1.��A�^v%+       ��K	v�-X��A�*

A2S/average_reward_1q�Au��+       ��K	`1X��A�*

A2S/average_reward_1%�^A-�Ŀ+       ��K	��<X��A�*

A2S/average_reward_1Q�B�g�C+       ��K	{�>X��A�*

A2S/average_reward_1vU�A5MFQ+       ��K	�4JX��A�*

A2S/average_reward_1��A��Q+       ��K	�HLX��A�*

A2S/average_reward_1t��A�++       ��K	"�PX��A�*

A2S/average_reward_1F��A��+       ��K		TX��A�*

A2S/average_reward_1���A;,�+       ��K	J�VX��A�*

A2S/average_reward_1��AU���+       ��K	��XX��A�*

A2S/average_reward_1�'�A~;�+       ��K	̄[X��A�*

A2S/average_reward_1�A�XQ�+       ��K	Y3`X��A�*

A2S/average_reward_1y��A��MM+       ��K	�<bX��A�*

A2S/average_reward_1���A��\�+       ��K	��dX��A�*

A2S/average_reward_1p��A5z2w       洦�		(��X��A�*i

A2S/kllY�:

A2S/policy_network_loss�z��

A2S/value_network_loss�B

A2S/q_network_loss}BY&�+       ��K	g#��X��A�*

A2S/average_reward_1|�Aws;0+       ��K	A��X��A�*

A2S/average_reward_1`J�AC@$+       ��K	�u��X��A�*

A2S/average_reward_1?��Azq�S+       ��K	u���X��A�*

A2S/average_reward_1�c�AS��+       ��K	Nd��X��A�*

A2S/average_reward_1��vA�$W�+       ��K	|�ÇX��A�*

A2S/average_reward_1���A`�+       ��K	�oǇX��A�*

A2S/average_reward_1��A�K�+       ��K	�|ˇX��A�*

A2S/average_reward_1��ASK+       ��K	�·X��A�*

A2S/average_reward_1�֙A#8��+       ��K	�чX��A�*

A2S/average_reward_1Me�A�B�2+       ��K	��ԇX��A�*

A2S/average_reward_1	�bA��%+       ��K	_هX��A�*

A2S/average_reward_1�G�A�� �+       ��K	Y�އX��A�*

A2S/average_reward_16:�A96�+       ��K	�,�X��A�*

A2S/average_reward_1)�A����+       ��K	�=�X��A�*

A2S/average_reward_1c�A��P�+       ��K	�G�X��A�*

A2S/average_reward_1n½AH�~+       ��K	�9�X��A�*

A2S/average_reward_1A7�#�+       ��K	�X��A�*

A2S/average_reward_1�l�A)��+       ��K	�(��X��A�*

A2S/average_reward_1��Az���+       ��K	7E��X��A�*

A2S/average_reward_1��A�p!�+       ��K	�e��X��A�*

A2S/average_reward_1�L�A=���+       ��K	k� �X��A�*

A2S/average_reward_19a�A��,+       ��K	6w�X��A�*

A2S/average_reward_1*�A�ޣ�+       ��K	���X��A�*

A2S/average_reward_1��A�e�+       ��K	���X��A�*

A2S/average_reward_1n2�A45��+       ��K	���X��A�*

A2S/average_reward_1�\�Ab>�n+       ��K	���X��A�*

A2S/average_reward_1�҇Ab�ʞ+       ��K	o��X��A�*

A2S/average_reward_1�f�A��
�+       ��K	��X��A�*

A2S/average_reward_1[s�A ��+       ��K	�9!�X��A�*

A2S/average_reward_1���Aj��+       ��K	1�$�X��A�*

A2S/average_reward_1���A2p��+       ��K	��&�X��A�*

A2S/average_reward_1��A���+       ��K	�*�X��A�*

A2S/average_reward_1���Ay�i�+       ��K	��-�X��A�*

A2S/average_reward_1?��A�i�+       ��K	�0�X��A�*

A2S/average_reward_1�߻A�r�+       ��K	�3�X��A�*

A2S/average_reward_1���A@n��+       ��K	6�X��A�*

A2S/average_reward_1R6�A�� �+       ��K	�9�X��A�*

A2S/average_reward_1�A����+       ��K	�;�X��A�*

A2S/average_reward_1�y|A��W9+       ��K	�w=�X��A�*

A2S/average_reward_1aɂA��+       ��K	�&@�X��A�*

A2S/average_reward_1�/�A1��+       ��K	��C�X��A�*

A2S/average_reward_1㏻A��+       ��K	� G�X��A�*

A2S/average_reward_1��A���+       ��K	��I�X��A�*

A2S/average_reward_1ِ�Aٕ�+       ��K	7�L�X��A�*

A2S/average_reward_1H�A#��+       ��K	�P�X��A�*

A2S/average_reward_1�
�A��C++       ��K	��Q�X��A�*

A2S/average_reward_1|A�|�+       ��K	o�S�X��A�*

A2S/average_reward_1*��A�W/+       ��K	�jV�X��A�*

A2S/average_reward_1n�A|�/+       ��K	�+Y�X��A�*

A2S/average_reward_1�y�A7��+       ��K	��\�X��A�*

A2S/average_reward_1���Az 4+       ��K	��^�X��A�*

A2S/average_reward_1��A9��D+       ��K	�Xc�X��A�*

A2S/average_reward_1}͸A`]�+       ��K	;f�X��A�*

A2S/average_reward_1��A`l�f+       ��K	Bh�X��A�*

A2S/average_reward_1���A*|)w       洦�	�+�X��A�*i

A2S/klvy�:

A2S/policy_network_lossOӿ

A2S/value_network_loss�]A

A2S/q_network_loss�VAw ��+       ��K	{�X��A�*

A2S/average_reward_1�±A�z+       ��K	&��X��A�*

A2S/average_reward_1 �A�ȧ�+       ��K	��X��A�*

A2S/average_reward_1��AC d�+       ��K	B��X��A�*

A2S/average_reward_1�A2���+       ��K	gY�X��A�*

A2S/average_reward_1L�A�Hο+       ��K	���X��A�*

A2S/average_reward_1���A��%+       ��K	����X��A�*

A2S/average_reward_1���A�x�+       ��K	�D��X��A�*

A2S/average_reward_1�ȨA�1L�+       ��K	����X��A�*

A2S/average_reward_1�c�Aj�[�+       ��K	��X��A�*

A2S/average_reward_1�ݓA��+�+       ��K	���X��A�*

A2S/average_reward_1���A�.ܹ+       ��K	k�	�X��A�*

A2S/average_reward_1ϑ�A�v�e+       ��K	#�X��A�*

A2S/average_reward_1ӵ�A���+       ��K	G"�X��A�*

A2S/average_reward_1̉A�.��+       ��K	-��X��A�*

A2S/average_reward_1s�cA<�.+       ��K	EA�X��A�*

A2S/average_reward_1���A��<�+       ��K	���X��A�*

A2S/average_reward_1q-�A��w4+       ��K	]��X��A�*

A2S/average_reward_1ΌA�+       ��K	�o�X��A�*

A2S/average_reward_1s|�Aٗd�+       ��K	���X��A�*

A2S/average_reward_1��A4�dW+       ��K	=� �X��A�*

A2S/average_reward_1��lA߲ژ+       ��K	`�#�X��A�*

A2S/average_reward_1q��A��K+       ��K	A�%�X��A�*

A2S/average_reward_1�͡AB�V'+       ��K	�q(�X��A�*

A2S/average_reward_1R��A5e+       ��K	h�+�X��A�*

A2S/average_reward_1e�AN�|�+       ��K	R�.�X��A�*

A2S/average_reward_1��AWt�+       ��K	�1�X��A�*

A2S/average_reward_1Wh�A�+       ��K	�:3�X��A�*

A2S/average_reward_1+M�A�|"{+       ��K	\5�X��A�*

A2S/average_reward_1�@zAP�/+       ��K	�7�X��A�*

A2S/average_reward_1�ΡA�L�J+       ��K	�/9�X��A�*

A2S/average_reward_1 �A�K�+       ��K	~=�X��A�*

A2S/average_reward_1�G�A�P{�+       ��K	��?�X��A�*

A2S/average_reward_1�m�A�?�m+       ��K	�D�X��A�*

A2S/average_reward_1~�A���S+       ��K	O�G�X��A�*

A2S/average_reward_1=��A�>��+       ��K	H�I�X��A�*

A2S/average_reward_1�ÌA�~�+       ��K	I�M�X��A�*

A2S/average_reward_1��AxY��+       ��K	1�P�X��A�*

A2S/average_reward_1&��A�D�+       ��K	~�S�X��A�*

A2S/average_reward_1���A.�6i+       ��K	��X�X��A�*

A2S/average_reward_1�A/R�+       ��K	�\�X��A�*

A2S/average_reward_1XW�Ar@�+       ��K	F�^�X��A�*

A2S/average_reward_16��A�HD�+       ��K	ܤa�X��A�*

A2S/average_reward_1�{�AC�eW+       ��K	z;c�X��A�*

A2S/average_reward_1�p�A?�r+       ��K	Ke�X��A�*

A2S/average_reward_1��SA�H��+       ��K	pg�X��A�*

A2S/average_reward_1ת�A��+       ��K	��i�X��A�*

A2S/average_reward_1��A!#�+       ��K	Ƅk�X��A�*

A2S/average_reward_1U
_AD(�+       ��K	��m�X��A�*

A2S/average_reward_1���A�A|`+       ��K	G�o�X��A�*

A2S/average_reward_1�X�Ao�M+       ��K	��r�X��A�*

A2S/average_reward_10#�A��>+       ��K	�t�X��A�*

A2S/average_reward_1���A���+       ��K	F�v�X��A�*

A2S/average_reward_13��AA%"�+       ��K	�cx�X��A�*

A2S/average_reward_1�F�A觥+       ��K	�sz�X��A�*

A2S/average_reward_1��A`n;�+       ��K	~"}�X��A�*

A2S/average_reward_1�I�A
�E�+       ��K	.e�X��A�*

A2S/average_reward_1닗A�(0+       ��K	�΁�X��A�*

A2S/average_reward_1���A�e�+       ��K	냑X��A�*

A2S/average_reward_1]p�A�)�+       ��K	��X��A�*

A2S/average_reward_1z�Az�+       ��K	+���X��A�*

A2S/average_reward_1��A��Z+       ��K	�L��X��A�*

A2S/average_reward_1�l�Ak��+       ��K	�;��X��A�*

A2S/average_reward_1���A�?2h+       ��K	�n��X��A�*

A2S/average_reward_1��A��"x+       ��K	(萑X��A�*

A2S/average_reward_1~��A6�e�+       ��K	I���X��A�*

A2S/average_reward_1@�A�v��+       ��K	7ؕ�X��A�*

A2S/average_reward_1I{A1�!w       洦�	�ۙX��A�*i

A2S/kl6�:

A2S/policy_network_loss��

A2S/value_network_loss��@

A2S/q_network_loss`*�@�8��+       ��K	�ߙX��A�*

A2S/average_reward_1fQ�A~�{ +       ��K	BQ�X��A�*

A2S/average_reward_1�"�A<.��+       ��K	1_�X��A�*

A2S/average_reward_1���A<t+       ��K	��X��A�*

A2S/average_reward_1k�A^*z�+       ��K	���X��A�*

A2S/average_reward_1�Z�Aon�+       ��K	)�X��A�*

A2S/average_reward_1�՞A�YRH+       ��K	� �X��A�*

A2S/average_reward_1r�A�
b+       ��K	3���X��A�*

A2S/average_reward_1��A��0+       ��K		��X��A�*

A2S/average_reward_1�A$���+       ��K	o���X��A�*

A2S/average_reward_1��A�Cb+       ��K	�Q��X��A�*

A2S/average_reward_1$��A���J+       ��K	�[��X��A�*

A2S/average_reward_1��A�f�+       ��K	ԫ�X��A�*

A2S/average_reward_1���A[ Z+       ��K	v��X��A�*

A2S/average_reward_1O�Ah�׾+       ��K	Z��X��A�*

A2S/average_reward_1�!�A$m�+       ��K	��
�X��A�*

A2S/average_reward_1�AY�^+       ��K	m%�X��A�*

A2S/average_reward_1=��Aʗ �+       ��K	���X��A�*

A2S/average_reward_1\7�AZ'�+       ��K	u9�X��A�*

A2S/average_reward_1_��A���?+       ��K	P��X��A�*

A2S/average_reward_1�-�A@U��+       ��K	2<�X��A�*

A2S/average_reward_1��A=6F+       ��K	�&�X��A�*

A2S/average_reward_1 ��A �n+       ��K	� �X��A�*

A2S/average_reward_1.�Ay��+       ��K	t�"�X��A�*

A2S/average_reward_1k�A�W�>+       ��K	t�%�X��A�*

A2S/average_reward_1Ѝ�AI�(+       ��K	<.)�X��A�*

A2S/average_reward_1�e�A���D+       ��K	|�+�X��A�*

A2S/average_reward_1B:�A�;v�+       ��K	�R-�X��A�*

A2S/average_reward_1N�Al��+       ��K	��/�X��A�*

A2S/average_reward_1ԫ�A�l�+       ��K	�A2�X��A�*

A2S/average_reward_1�Y�A�/�+       ��K	B�4�X��A�*

A2S/average_reward_1��Ad�+       ��K	�7�X��A�*

A2S/average_reward_1�D�A|�+       ��K	
�;�X��A�*

A2S/average_reward_1�ߌAtFT+       ��K	ʫ>�X��A�*

A2S/average_reward_1屩A�y"q+       ��K	��@�X��A�*

A2S/average_reward_1�=�A k�+       ��K	P�C�X��A�*

A2S/average_reward_1��A̪��+       ��K	�dE�X��A�*

A2S/average_reward_1�y�A�V+       ��K	�G�X��A�*

A2S/average_reward_1�|A�C+       ��K	��J�X��A�*

A2S/average_reward_1���AV��a+       ��K	5�M�X��A�*

A2S/average_reward_1!�A(h�+       ��K	>�P�X��A�*

A2S/average_reward_1R�A0 �`+       ��K	��R�X��A�*

A2S/average_reward_1Qa�AܰK�+       ��K	�<V�X��A�*

A2S/average_reward_1GݽAɹ�b+       ��K	�EY�X��A�*

A2S/average_reward_14�A�Q�+       ��K	��]�X��A�*

A2S/average_reward_1y��AS냕+       ��K	�Db�X��A�*

A2S/average_reward_1���A��N+       ��K	�Qe�X��A�*

A2S/average_reward_1j�AL�2'+       ��K	�Ji�X��A�*

A2S/average_reward_1�?�As�+       ��K	�fm�X��A�*

A2S/average_reward_1+��A����+       ��K	�9s�X��A�*

A2S/average_reward_1���Aޥ1�+       ��K	>�v�X��A�*

A2S/average_reward_1JV�A�� `+       ��K	�[|�X��A�*

A2S/average_reward_1$L�AD�X]+       ��K	�8�X��A�*

A2S/average_reward_1!�AɌ=e+       ��K	� ��X��A�*

A2S/average_reward_1�@�A"�Yr+       ��K	X���X��A�*

A2S/average_reward_1|�Aؤ�+       ��K	圉�X��A�*

A2S/average_reward_1���A�x+       ��K	���X��A�*

A2S/average_reward_1\�A��+       ��K	�\��X��A�*

A2S/average_reward_19��A2 �+       ��K	����X��A�*

A2S/average_reward_1gߦAR�/+       ��K	-ؓ�X��A�*

A2S/average_reward_1�.�A�'��+       ��K	e���X��A�*

A2S/average_reward_1�ڐA�_��+       ��K	Q���X��A�*

A2S/average_reward_1N�A~r�+       ��K	,���X��A�*

A2S/average_reward_1�8�A��+       ��K	\r��X��A�*

A2S/average_reward_1�O�A�'�w       洦�	�� �X��A�*i

A2S/kl���:

A2S/policy_network_loss�?ӿ

A2S/value_network_lossX�@

A2S/q_network_lossl�@�{eE+       ��K	$�X��A�*

A2S/average_reward_1y�A��~�+       ��K	A`�X��A�*

A2S/average_reward_14A�A�N�+       ��K	U�X��A�*

A2S/average_reward_1���A>�@�+       ��K	���X��A�*

A2S/average_reward_1��A/�+       ��K	Ձ�X��A�*

A2S/average_reward_1 ��Ai��)+       ��K	E`�X��A�*

A2S/average_reward_1��A�W)�+       ��K	�V�X��A�*

A2S/average_reward_1G�A�ɦ+       ��K	�9 �X��A�*

A2S/average_reward_1z͂A���6+       ��K	u/#�X��A�*

A2S/average_reward_1��AL�r�+       ��K	�	&�X��A�*

A2S/average_reward_1�AL-�+       ��K	7�(�X��A�*

A2S/average_reward_1+�A�@ku+       ��K	*4,�X��A�*

A2S/average_reward_18��A����+       ��K	�e0�X��A�*

A2S/average_reward_1Ƅ�Ad��i+       ��K	oF3�X��A�*

A2S/average_reward_1���A\D��+       ��K	��7�X��A�*

A2S/average_reward_1Q�A�_*+       ��K	� <�X��A�*

A2S/average_reward_1e#�A�T�+       ��K	}�?�X��A�*

A2S/average_reward_1�~�A�R^�+       ��K	�2D�X��A�*

A2S/average_reward_1[ĹAq|�+       ��K	��H�X��A�*

A2S/average_reward_1°�A�^�+       ��K	fAN�X��A�*

A2S/average_reward_1�@�A��(+       ��K	w#P�X��A�*

A2S/average_reward_1���A'+       ��K	�TR�X��A�*

A2S/average_reward_1�ҐA��w+       ��K	��U�X��A�*

A2S/average_reward_1`߲AWI̜+       ��K	�=Z�X��A�*

A2S/average_reward_1U��A��ތ+       ��K	��]�X��A�*

A2S/average_reward_1nn�AH_��+       ��K	��_�X��A�*

A2S/average_reward_1p��An�f�+       ��K	��c�X��A�*

A2S/average_reward_1r�AU�+       ��K	�f�X��A�*

A2S/average_reward_1�L�A���+       ��K	��h�X��A�*

A2S/average_reward_1�"mA=��+       ��K	�gk�X��A�*

A2S/average_reward_1���A�+       ��K	:7n�X��A�*

A2S/average_reward_1��AYj�)+       ��K	�mp�X��A�*

A2S/average_reward_1��A�y��+       ��K	�Zu�X��A�*

A2S/average_reward_1Y��A��+       ��K	]x�X��A�*

A2S/average_reward_1 ��A�Y�+       ��K	*�z�X��A�*

A2S/average_reward_12��A���+       ��K	�G~�X��A�*

A2S/average_reward_1{h�Aj�0+       ��K	�À�X��A�*

A2S/average_reward_1JޜA3��W+       ��K	q	��X��A�*

A2S/average_reward_1&ޫA���+       ��K	
��X��A�*

A2S/average_reward_1�A3�U+       ��K	�&��X��A�*

A2S/average_reward_1"�A���+       ��K	p��X��A�*

A2S/average_reward_1jݿAQs��+       ��K	�z��X��A�*

A2S/average_reward_1���A�rH+       ��K	�O��X��A�*

A2S/average_reward_1J�A���t+       ��K	�5��X��A�*

A2S/average_reward_1��A�'��+       ��K	_��X��A�*

A2S/average_reward_1�ɒA�4+       ��K	�Y��X��A�*

A2S/average_reward_1�hA8k�+       ��K	�p��X��A�*

A2S/average_reward_1�Z�Aۍ�	+       ��K	���X��A�*

A2S/average_reward_1�Az���+       ��K	�X��A�*

A2S/average_reward_1�A�j��+       ��K	i���X��A�*

A2S/average_reward_1[6�Ab��w+       ��K	E��X��A�*

A2S/average_reward_1ۊA�&�+       ��K	o���X��A�*

A2S/average_reward_1�.�A�Vw       洦�	����X��A�*i

A2S/kl���:

A2S/policy_network_lossbf��

A2S/value_network_loss�A

A2S/q_network_loss���@,�+       ��K	fK��X��A�*

A2S/average_reward_1皝A+˼g+       ��K	�9�X��A�*

A2S/average_reward_1�p�A)��+       ��K	P��X��A� *

A2S/average_reward_1$2�A*�%�+       ��K	U��X��A� *

A2S/average_reward_1o��Af;a�+       ��K	�(�X��A� *

A2S/average_reward_1�בAL��+       ��K	���X��A� *

A2S/average_reward_1�]�A��P�+       ��K	���X��A� *

A2S/average_reward_1$�A�fv}+       ��K	C��X��A� *

A2S/average_reward_1�7�A�2?+       ��K	� �X��A� *

A2S/average_reward_1L��A0L�Q+       ��K	��"�X��A� *

A2S/average_reward_1�|�A�F�E+       ��K	6�&�X��A� *

A2S/average_reward_1P�Aa��+       ��K		`*�X��A� *

A2S/average_reward_12��Ajgp=+       ��K	p�/�X��A� *

A2S/average_reward_1D&�A;q�b+       ��K	G3�X��A� *

A2S/average_reward_1Ë�A{	,+       ��K	Ԥ6�X��A�!*

A2S/average_reward_1��A>��i+       ��K	��9�X��A�!*

A2S/average_reward_1'��A�қ+       ��K	��A�X��A�!*

A2S/average_reward_1iZ�A�_�{+       ��K	��E�X��A�!*

A2S/average_reward_1T��A%!4�+       ��K	|�H�X��A�!*

A2S/average_reward_1=��A����+       ��K	-�K�X��A�!*

A2S/average_reward_1H�Atj<�+       ��K	�PS�X��A�!*

A2S/average_reward_1sH�A���~+       ��K	�W�X��A�!*

A2S/average_reward_1��A;��t+       ��K	U�^�X��A�!*

A2S/average_reward_1c��A��0Q+       ��K	W�e�X��A�!*

A2S/average_reward_1�k�A�@+       ��K	3�j�X��A�!*

A2S/average_reward_1<�Ad���+       ��K	om�X��A�"*

A2S/average_reward_1���Ag8�+       ��K	cq�X��A�"*

A2S/average_reward_1o)�A��8++       ��K	v�s�X��A�"*

A2S/average_reward_1b�A ��'+       ��K	�7v�X��A�"*

A2S/average_reward_1k|�A&_o+       ��K	�w�X��A�"*

A2S/average_reward_1��A25��+       ��K	��y�X��A�"*

A2S/average_reward_1ǢAc+@+       ��K	��}�X��A�"*

A2S/average_reward_1o��A�X��+       ��K	�C��X��A�"*

A2S/average_reward_1�ߚAxas1+       ��K	o��X��A�"*

A2S/average_reward_1a �A(5�W+       ��K	c���X��A�"*

A2S/average_reward_1P�A�W�+       ��K	p���X��A�"*

A2S/average_reward_1ՖATӊ�+       ��K	y@��X��A�"*

A2S/average_reward_1���A�Q�E+       ��K	ݏ��X��A�"*

A2S/average_reward_11��A�pL+       ��K	Qܓ�X��A�"*

A2S/average_reward_1�g�A���+       ��K	�p��X��A�#*

A2S/average_reward_1?��A쨺�+       ��K	b��X��A�#*

A2S/average_reward_1�4�A����+       ��K	����X��A�#*

A2S/average_reward_1-z�A�qR{+       ��K	����X��A�#*

A2S/average_reward_1r�AɈ�U+       ��K	�砬X��A�#*

A2S/average_reward_1�]�A|Э+       ��K	c���X��A�#*

A2S/average_reward_1��A&��%+       ��K	8���X��A�#*

A2S/average_reward_1��Ax�W�+       ��K	�ī�X��A�#*

A2S/average_reward_1�5�AE�[�+       ��K	�t��X��A�#*

A2S/average_reward_1IٴARY+       ��K	蕲�X��A�#*

A2S/average_reward_1�-�A�1+       ��K	����X��A�#*

A2S/average_reward_1~��A�نiw       洦�	^��X��A�#*i

A2S/kl���:

A2S/policy_network_lossE��

A2S/value_network_loss"�0A

A2S/q_network_losswf�@�ƫ�+       ��K	~
!�X��A�#*

A2S/average_reward_1a(�Am>M+       ��K	�Y$�X��A�#*

A2S/average_reward_1C��AF\�+       ��K	?�'�X��A�#*

A2S/average_reward_1q�AJ��+       ��K	�-,�X��A�$*

A2S/average_reward_1���A>�Z+       ��K	��0�X��A�$*

A2S/average_reward_1O�AMw
:+       ��K	��4�X��A�$*

A2S/average_reward_1H;�A�K�3+       ��K	d�7�X��A�$*

A2S/average_reward_17R�Au�O+       ��K	��=�X��A�$*

A2S/average_reward_1m�A�n��+       ��K	1�@�X��A�$*

A2S/average_reward_1vܢAH�P�+       ��K	��C�X��A�$*

A2S/average_reward_1�A:��x+       ��K	�eG�X��A�$*

A2S/average_reward_1��A�)�+       ��K	g�L�X��A�$*

A2S/average_reward_1��AI�J8+       ��K	�hP�X��A�$*

A2S/average_reward_1�Ag��+       ��K	��S�X��A�%*

A2S/average_reward_1<ۿAG�Y+       ��K	I'X�X��A�%*

A2S/average_reward_1�E�A͕+       ��K	�[�X��A�%*

A2S/average_reward_1��Aaa�+       ��K	��_�X��A�%*

A2S/average_reward_1���A]���+       ��K	� d�X��A�%*

A2S/average_reward_1�;�A]a��+       ��K	� i�X��A�%*

A2S/average_reward_1���AmU4�+       ��K	�n�X��A�%*

A2S/average_reward_1�Y�Af]�7+       ��K	�u�X��A�%*

A2S/average_reward_1u��A�Wr+       ��K	΁z�X��A�%*

A2S/average_reward_1\@�A�֦+       ��K	��X��A�&*

A2S/average_reward_1#w�A���+       ��K	c��X��A�&*

A2S/average_reward_1E��A��[+       ��K	ѧ��X��A�&*

A2S/average_reward_1d�AY�+       ��K	����X��A�&*

A2S/average_reward_1��Ak�ch+       ��K	:��X��A�&*

A2S/average_reward_1��A��e�+       ��K	 ד�X��A�&*

A2S/average_reward_1���A��
�+       ��K	���X��A�&*

A2S/average_reward_1�5�Aۄ��+       ��K	���X��A�&*

A2S/average_reward_1���A�?�+       ��K	9ꢵX��A�&*

A2S/average_reward_1�:B��NH+       ��K	c>��X��A�&*

A2S/average_reward_1�XAxx �+       ��K	(�X��A�&*

A2S/average_reward_1�V�Aķ�K+       ��K	����X��A�&*

A2S/average_reward_1��A�Y�<+       ��K	𹱵X��A�'*

A2S/average_reward_1��Ai�vD+       ��K	_���X��A�'*

A2S/average_reward_1��Bt�Yp+       ��K	PD��X��A�'*

A2S/average_reward_1-^�AJ�A+       ��K	�`��X��A�'*

A2S/average_reward_1��A��Ô+       ��K	#���X��A�'*

A2S/average_reward_1;õA-��7+       ��K	=ȵX��A�'*

A2S/average_reward_1��A;��`+       ��K	xt̵X��A�'*

A2S/average_reward_1O=�A|K_�w       洦�	8�X��A�'*i

A2S/kl���:

A2S/policy_network_loss򐲿

A2S/value_network_loss��A

A2S/q_network_loss��MAP��c+       ��K	j�=�X��A�'*

A2S/average_reward_1���A{��+       ��K	��A�X��A�'*

A2S/average_reward_1'��A��^+       ��K	�IH�X��A�'*

A2S/average_reward_1Ik�Au��l+       ��K	��K�X��A�'*

A2S/average_reward_1=�A�}�+       ��K	�(Q�X��A�(*

A2S/average_reward_1��A�A��+       ��K	�8U�X��A�(*

A2S/average_reward_1�нA��?+       ��K	dg[�X��A�(*

A2S/average_reward_1��A�Z�_+       ��K	�rb�X��A�(*

A2S/average_reward_1�=�A7�$T+       ��K	Qh�X��A�(*

A2S/average_reward_1��AzK3�+       ��K	�
k�X��A�(*

A2S/average_reward_1u'�A2�Bh+       ��K	�n�X��A�(*

A2S/average_reward_1a
�A�^3�+       ��K	�r�X��A�(*

A2S/average_reward_1{��A�@!�+       ��K	R2{�X��A�(*

A2S/average_reward_1��A��+       ��K	�j��X��A�(*

A2S/average_reward_1�7�AV�P0+       ��K	����X��A�)*

A2S/average_reward_1���A�D��+       ��K	����X��A�)*

A2S/average_reward_1�Y�AisB�+       ��K	��X��A�)*

A2S/average_reward_1�3�A��e�+       ��K	SU��X��A�)*

A2S/average_reward_1O �A*�+       ��K	ku��X��A�)*

A2S/average_reward_1��A���+       ��K	5��X��A�)*

A2S/average_reward_1��A�|�+       ��K	~밾X��A�)*

A2S/average_reward_1P��A����+       ��K	,:��X��A�)*

A2S/average_reward_1z�A�f�+       ��K	�6��X��A�)*

A2S/average_reward_1>ĬATdD�+       ��K	u�¾X��A�**

A2S/average_reward_1�BP� +       ��K	�ȾX��A�**

A2S/average_reward_1�L�A����+       ��K	;;X��A�**

A2S/average_reward_1Z��AM-c$+       ��K	dnѾX��A�**

A2S/average_reward_1��A��B+       ��K	rP׾X��A�**

A2S/average_reward_1���A�F1+       ��K	�۾X��A�**

A2S/average_reward_1�Ai���+       ��K	��X��A�**

A2S/average_reward_1k�AZ-��+       ��K	���X��A�**

A2S/average_reward_1��A���+       ��K	Ƽ�X��A�**

A2S/average_reward_1r�A�x�+       ��K	E_��X��A�+*

A2S/average_reward_1��A��fc+       ��K	Z��X��A�+*

A2S/average_reward_1���A&�+       ��K	����X��A�+*

A2S/average_reward_1��A�A�+       ��K	KG�X��A�+*

A2S/average_reward_1�A=�]*+       ��K	�	�X��A�+*

A2S/average_reward_1xԮA��i:+       ��K	� �X��A�+*

A2S/average_reward_1�^�An�Lw       洦�	=�E�X��A�+*i

A2S/kl�5;

A2S/policy_network_loss����

A2S/value_network_loss�fA

A2S/q_network_loss`�
Ae��'+       ��K	c�O�X��A�+*

A2S/average_reward_1���A�a�p+       ��K	�8U�X��A�+*

A2S/average_reward_1f�A���+       ��K	>Z�X��A�+*

A2S/average_reward_1�R�A����+       ��K	��b�X��A�,*

A2S/average_reward_1x�B�0h+       ��K	�h�X��A�,*

A2S/average_reward_1A��A�a	+       ��K	QOm�X��A�,*

A2S/average_reward_1Ŋ�AK��+       ��K	��s�X��A�,*

A2S/average_reward_1���A�۳+       ��K	B�|�X��A�,*

A2S/average_reward_1m�B�a�+       ��K	���X��A�,*

A2S/average_reward_1r�AB�٘+       ��K	z���X��A�,*

A2S/average_reward_1Fl�A�~�,+       ��K	*���X��A�,*

A2S/average_reward_18��A�=pp+       ��K	�ޕ�X��A�-*

A2S/average_reward_19�A���X+       ��K	�ƛ�X��A�-*

A2S/average_reward_1�\�A}���+       ��K	�R��X��A�-*

A2S/average_reward_1�AL��+       ��K	���X��A�-*

A2S/average_reward_1̝�A��+       ��K	墫�X��A�-*

A2S/average_reward_1�M�AK��++       ��K	�S��X��A�-*

A2S/average_reward_1�|�A�
+       ��K	�7��X��A�-*

A2S/average_reward_1d��A{��+       ��K	����X��A�-*

A2S/average_reward_1gk�Ay�dH+       ��K	���X��A�-*

A2S/average_reward_1|�A����+       ��K	���X��A�.*

A2S/average_reward_1X��A4
 �+       ��K	]��X��A�.*

A2S/average_reward_1�ɼA����+       ��K	���X��A�.*

A2S/average_reward_1)�A-���+       ��K	(5��X��A�.*

A2S/average_reward_1W_�A@(�	+       ��K	�{��X��A�.*

A2S/average_reward_1��AY�(+       ��K	W/��X��A�.*

A2S/average_reward_1�Br4�+       ��K	l���X��A�.*

A2S/average_reward_1���AV+[�+       ��K	9���X��A�.*

A2S/average_reward_1&�B�ʅ+       ��K	����X��A�.*

A2S/average_reward_1@�A�H�+       ��K	^���X��A�.*

A2S/average_reward_1�ܛAR�9�+       ��K	��X��A�/*

A2S/average_reward_1�y�A�&�r+       ��K	��	�X��A�/*

A2S/average_reward_1?��A� `�+       ��K	�o�X��A�/*

A2S/average_reward_1��Ak���+       ��K	��X��A�/*

A2S/average_reward_1���A��0+       ��K	���X��A�/*

A2S/average_reward_1J��At2��w       洦�	q��X��A�/*i

A2S/kl/;

A2S/policy_network_lossmɲ�

A2S/value_network_loss���A

A2S/q_network_losspA& a�+       ��K	#��X��A�/*

A2S/average_reward_1���A�v0�+       ��K	�ޖ�X��A�/*

A2S/average_reward_1_.�A��/ +       ��K	����X��A�/*

A2S/average_reward_1,��A��1�+       ��K	�ß�X��A�0*

A2S/average_reward_1���A��+�+       ��K	�\��X��A�0*

A2S/average_reward_1'��A,���+       ��K	#%��X��A�0*

A2S/average_reward_1�6�AO���+       ��K	][��X��A�0*

A2S/average_reward_1���AwlD�+       ��K	<#��X��A�0*

A2S/average_reward_1�C�AQ�X+       ��K	�u��X��A�0*

A2S/average_reward_1c �Al�*+       ��K	����X��A�1*

A2S/average_reward_1n1A�Y�.+       ��K	X���X��A�1*

A2S/average_reward_1���Ah�a�+       ��K	���X��A�1*

A2S/average_reward_1���A����+       ��K	!V��X��A�1*

A2S/average_reward_1��A��Z�+       ��K	�'��X��A�1*

A2S/average_reward_1�"�A�.kS+       ��K	]k��X��A�1*

A2S/average_reward_1���A��i+       ��K	���X��A�1*

A2S/average_reward_1w�A�O{+       ��K	�v��X��A�2*

A2S/average_reward_1���A�1R*+       ��K	:W�X��A�2*

A2S/average_reward_1���A�u�+       ��K	�}
�X��A�2*

A2S/average_reward_1�A�A�`+       ��K	�|�X��A�2*

A2S/average_reward_1���A�� +       ��K	��X��A�2*

A2S/average_reward_1�j�A��\+       ��K	m� �X��A�2*

A2S/average_reward_1K�AD��+       ��K	d&�X��A�2*

A2S/average_reward_1��AY��\+       ��K	#!1�X��A�3*

A2S/average_reward_1��A{C��+       ��K	�y7�X��A�3*

A2S/average_reward_1��A2�!�+       ��K	�]<�X��A�3*

A2S/average_reward_1�K�A�G��+       ��K	7�D�X��A�3*

A2S/average_reward_1��BV*6�w       洦�	v��X��A�3*i

A2S/kl�;

A2S/policy_network_loss)���

A2S/value_network_loss�ݤA

A2S/q_network_loss�&yA4t�+       ��K	���X��A�3*

A2S/average_reward_1P��A@?%�+       ��K	����X��A�3*

A2S/average_reward_1>"�A����+       ��K	�4��X��A�3*

A2S/average_reward_1	a�A-:v+       ��K	�t��X��A�3*

A2S/average_reward_1RQ�A��u�+       ��K	 ��X��A�4*

A2S/average_reward_1�O�AKgX8+       ��K	W��X��A�4*

A2S/average_reward_1h�A%��+       ��K	����X��A�4*

A2S/average_reward_1��A+���+       ��K	&ߵ�X��A�4*

A2S/average_reward_1
&�AE�%+       ��K	���X��A�4*

A2S/average_reward_1�߃Aj���+       ��K	 ��X��A�4*

A2S/average_reward_1X!�A��lm+       ��K	����X��A�4*

A2S/average_reward_1�,�A*p�+       ��K	�Q��X��A�4*

A2S/average_reward_1©Bފ��+       ��K	���X��A�4*

A2S/average_reward_1?$�AA��+       ��K	a ��X��A�5*

A2S/average_reward_1�˼A���+       ��K	l{��X��A�5*

A2S/average_reward_1^��A�s +       ��K	�-��X��A�5*

A2S/average_reward_1*H�A�`��+       ��K	Z���X��A�5*

A2S/average_reward_1��A�y�W+       ��K	���X��A�5*

A2S/average_reward_1#�
B+=[�+       ��K	�u��X��A�5*

A2S/average_reward_1N��A0/��+       ��K	3���X��A�5*

A2S/average_reward_1ƛ�A��+       ��K	���X��A�5*

A2S/average_reward_1O_�A���_+       ��K	����X��A�6*

A2S/average_reward_1LYB{���+       ��K	��X��A�6*

A2S/average_reward_1I��A��@$+       ��K	x&�X��A�6*

A2S/average_reward_1���A)��t+       ��K	i�X��A�6*

A2S/average_reward_1���AL
it+       ��K	'N�X��A�6*

A2S/average_reward_1�B��8a+       ��K	ԙ�X��A�6*

A2S/average_reward_1�)�A}�w�+       ��K	��X��A�6*

A2S/average_reward_1v�A�>+       ��K	>�$�X��A�7*

A2S/average_reward_1և	B�_]+       ��K	��*�X��A�7*

A2S/average_reward_1o��A	F�7+       ��K	�/�X��A�7*

A2S/average_reward_1��A�m�=+       ��K	��3�X��A�7*

A2S/average_reward_1���A��"Vw       洦�	�6{�X��A�7*i

A2S/kl�v4;

A2S/policy_network_loss��޿

A2S/value_network_lossi-pA

A2S/q_network_loss
"A��E+       ��K	Aˀ�X��A�7*

A2S/average_reward_1���A?ǩ+       ��K	���X��A�7*

A2S/average_reward_1���A$��+       ��K	���X��A�7*

A2S/average_reward_1���A�ӕh+       ��K	�z��X��A�7*

A2S/average_reward_1~'�A��[�+       ��K	6ߒ�X��A�7*

A2S/average_reward_13�A2��O+       ��K	mt��X��A�7*

A2S/average_reward_1���A�
U+       ��K	�ښ�X��A�8*

A2S/average_reward_1�?�A��3+       ��K	C���X��A�8*

A2S/average_reward_1���A
���+       ��K	`A��X��A�8*

A2S/average_reward_1�s�AN�s�+       ��K	C���X��A�8*

A2S/average_reward_1�Bj�p+       ��K	T��X��A�8*

A2S/average_reward_1E<�A��u+       ��K	�	��X��A�8*

A2S/average_reward_1�/ B�K��+       ��K	A��X��A�8*

A2S/average_reward_1��A�I�N+       ��K	վ�X��A�8*

A2S/average_reward_1�v�Aû�A+       ��K	X��X��A�9*

A2S/average_reward_1���A.;��+       ��K	����X��A�9*

A2S/average_reward_1A�A]��[+       ��K	���X��A�9*

A2S/average_reward_1��B����+       ��K	���X��A�9*

A2S/average_reward_1�R�A��S�+       ��K	\���X��A�9*

A2S/average_reward_1�A��Z�+       ��K	"���X��A�9*

A2S/average_reward_1�[�A� �D+       ��K	f&��X��A�9*

A2S/average_reward_1>Bq��+       ��K	�4��X��A�:*

A2S/average_reward_1ze�A|�Z�+       ��K	o��X��A�:*

A2S/average_reward_1�?�A@\1i+       ��K	���X��A�:*

A2S/average_reward_1��A�f�+       ��K	l�X��A�:*

A2S/average_reward_1��A�n:E+       ��K	A��X��A�:*

A2S/average_reward_1_��Aq�X+       ��K	dx�X��A�:*

A2S/average_reward_1h��ArT�8+       ��K	j �X��A�:*

A2S/average_reward_16��A�aO+       ��K	�P$�X��A�:*

A2S/average_reward_1�_�A�J{+       ��K	�O.�X��A�:*

A2S/average_reward_1�;	B۰�+       ��K	 �4�X��A�;*

A2S/average_reward_1�V�A`uߐ+       ��K	w;�X��A�;*

A2S/average_reward_1g�A�e"�+       ��K	� D�X��A�;*

A2S/average_reward_1z�B�C&Rw       洦�	���X��A�;*i

A2S/kl҇T;

A2S/policy_network_lossn�Ϳ

A2S/value_network_loss��<A

A2S/q_network_loss���@����+       ��K	����X��A�;*

A2S/average_reward_1���AI�d�+       ��K	����X��A�;*

A2S/average_reward_1 wB�cm+       ��K	���X��A�;*

A2S/average_reward_1E�A���+       ��K	���X��A�<*

A2S/average_reward_1B�Bg�Є+       ��K	�s��X��A�<*

A2S/average_reward_1oB$"��+       ��K	@���X��A�<*

A2S/average_reward_1�m�A�".n+       ��K	-���X��A�<*

A2S/average_reward_1;�A�2v*+       ��K	³�X��A�<*

A2S/average_reward_1᮵A�2"�+       ��K	���X��A�<*

A2S/average_reward_1���A����+       ��K	���X��A�<*

A2S/average_reward_1X��AV��+       ��K	i� �X��A�<*

A2S/average_reward_1'f�A+��3+       ��K	S�'�X��A�=*

A2S/average_reward_16��AHK7+       ��K	��2�X��A�=*

A2S/average_reward_1Ò�A�ي�+       ��K	�b8�X��A�=*

A2S/average_reward_1Hm�A��ֱ+       ��K	��N�X��A�=*

A2S/average_reward_1�zqB�m�H+       ��K	�	\�X��A�=*

A2S/average_reward_1g�B�O5+       ��K	�f�X��A�>*

A2S/average_reward_1���A滽�+       ��K	�Xl�X��A�>*

A2S/average_reward_1#��A��O�+       ��K	�t�X��A�>*

A2S/average_reward_1�I�Ab��]+       ��K	���X��A�>*

A2S/average_reward_1["�A�E
+       ��K	�_��X��A�>*

A2S/average_reward_1&V
B�Ů
+       ��K	�D��X��A�>*

A2S/average_reward_1�'B詯)+       ��K	�F��X��A�?*

A2S/average_reward_1`9�A�N��+       ��K	=���X��A�?*

A2S/average_reward_1NO�Aĺ'+       ��K	Q���X��A�?*

A2S/average_reward_1�"B'�Zw       洦�	.��X��A�?*i

A2S/kl��&;

A2S/policy_network_loss�ӿ

A2S/value_network_loss���A

A2S/q_network_loss��A���C+       ��K	� �X��A�?*

A2S/average_reward_1T�B^�r@+       ��K	�=%�X��A�?*

A2S/average_reward_1E&�A-���+       ��K	9�X��A�@*

A2S/average_reward_1��IB^� y+       ��K	��>�X��A�@*

A2S/average_reward_1z6�A���u+       ��K	,�I�X��A�@*

A2S/average_reward_1���AIC�+       ��K	�R�X��A�@*

A2S/average_reward_1�F�A�y+       ��K	��]�X��A�A*

A2S/average_reward_1YB�w}�+       ��K	�"h�X��A�A*

A2S/average_reward_1��)B���6+       ��K	�m�X��A�A*

A2S/average_reward_1���AF��+       ��K	�Pv�X��A�A*

A2S/average_reward_1D�B�R�?+       ��K	j�~�X��A�A*

A2S/average_reward_1�FB+       ��K	���X��A�B*

A2S/average_reward_1�:B,��U+       ��K	H��X��A�B*

A2S/average_reward_1���A���+       ��K	ȝ�X��A�B*

A2S/average_reward_1��B&Vi+       ��K	+��X��A�B*

A2S/average_reward_1��AM�e�+       ��K	����X��A�B*

A2S/average_reward_1�F0B����+       ��K	$T��X��A�C*

A2S/average_reward_1���A�)�+       ��K	����X��A�C*

A2S/average_reward_1p�9BaP��w       洦�	91�X��A�C*i

A2S/kl��1;

A2S/policy_network_loss���

A2S/value_network_lossy��A

A2S/q_network_loss�j�A�h�+       ��K	 5>�X��A�C*

A2S/average_reward_1m��A�+��+       ��K	��\�X��A�D*

A2S/average_reward_1�QvBĮ��+       ��K	F^o�X��A�D*

A2S/average_reward_1V�_B�+m�+       ��K	��w�X��A�D*

A2S/average_reward_1�AL[�E+       ��K	w��X��A�E*

A2S/average_reward_1�O�A&!+       ��K	dS��X��A�E*

A2S/average_reward_19aBtmt+       ��K	4���X��A�F*

A2S/average_reward_1���B��+       ��K	���X��A�F*

A2S/average_reward_1j��A��*+       ��K	�@��X��A�F*

A2S/average_reward_1l[�A��T�+       ��K	>��X��A�G*

A2S/average_reward_1��>B���w       洦�	E�cY��A�G*i

A2S/klK0;

A2S/policy_network_loss���

A2S/value_network_loss:�B

A2S/q_network_loss�0uB~�Z+       ��K	�fxY��A�G*

A2S/average_reward_1Պ BJC��+       ��K	���Y��A�H*

A2S/average_reward_1�cA;� +       ��K	��Y��A�H*

A2S/average_reward_1�tB�z�+       ��K	�ĘY��A�H*

A2S/average_reward_17�A/�7+       ��K	�Y��A�H*

A2S/average_reward_1y?B
G��+       ��K	���Y��A�I*

A2S/average_reward_1�aBL5��+       ��K	R��Y��A�J*

A2S/average_reward_1kn�BS�^+       ��K	�Y��A�J*

A2S/average_reward_1��Bq���+       ��K	^�!Y��A�K*

A2S/average_reward_1b�A!�_X+       ��K	`�*Y��A�K*

A2S/average_reward_1��B�|�w       洦�	�iY��A�K*i

A2S/kl��(;

A2S/policy_network_loss��

A2S/value_network_loss_��B

A2S/q_network_loss��B񧘪+       ��K	��sY��A�K*

A2S/average_reward_17AB��45+       ��K	?^�Y��A�L*

A2S/average_reward_1�8
B9s��+       ��K	;��Y��A�L*

A2S/average_reward_1�OgBt��+       ��K	��Y��A�M*

A2S/average_reward_1&��Bf3�+       ��K	���Y��A�N*

A2S/average_reward_1#�nB�@M+       ��K	#�Y��A�O*

A2S/average_reward_1A��B�+Z	+       ��K	h�Y��A�O*

A2S/average_reward_1��B{��?w       洦�	E��Y��A�O*i

A2S/kl�� ;

A2S/policy_network_loss�6��

A2S/value_network_lossS27C

A2S/q_network_loss^� Ca���+       ��K	#�Y��A�Q*

A2S/average_reward_1�E�B�m+       ��K	�lY��A�S*

A2S/average_reward_1�]CW�w       洦�	��"Y��A�S*i

A2S/kl�.;

A2S/policy_network_loss -u�

A2S/value_network_loss�<LC

A2S/q_network_loss�3CEEyX+       ��K	Q��"Y��A�T*

A2S/average_reward_1�ADFD?+       ��K	)L�"Y��A�T*

A2S/average_reward_1:�B�B!�+       ��K	���"Y��A�U*

A2S/average_reward_17*�B�PlA+       ��K	�(X#Y��A�V*

A2S/average_reward_1O�!C�O)+       ��K	��#Y��A�X*

A2S/average_reward_1r�C���w       洦�	�L�+Y��A�X*i

A2S/klr8;

A2S/policy_network_loss�ӭ�

A2S/value_network_loss*�C

A2S/q_network_lossVCe$�+       ��K	uj,Y��A�[*

A2S/average_reward_1ߒxC?��+       ��K	�|,Y��A�[*

A2S/average_reward_1��iB�d7�+       ��K	�ى,Y��A�\*

A2S/average_reward_1[T B�s+       ��K	��,Y��A�\*

A2S/average_reward_1cPA��z�+       ��K	��,Y��A�^*

A2S/average_reward_1#�3C�O��w       洦�	N�j5Y��A�^*i

A2S/kl{�;

A2S/policy_network_lossD��

A2S/value_network_loss�C

A2S/q_network_loss��C���$+       ��K	(55Y��A�^*

A2S/average_reward_1�%B��'�+       ��K	�{�5Y��A�`*

A2S/average_reward_1�h-C���+       ��K	NC6Y��A�b*

A2S/average_reward_1ؚ�B�d��+       ��K	<�6Y��A�b*

A2S/average_reward_19#B_u�Gw       洦�	�g>Y��A�b*i

A2S/kl`m�;

A2S/policy_network_loss����

A2S/value_network_losso (C

A2S/q_network_loss\Cs��+       ��K	���>Y��A�e*

A2S/average_reward_1P�rC���+       ��K	qq�?Y��A�i*

A2S/average_reward_1��C�8�w       洦�	��2HY��A�i*i

A2S/kl���:

A2S/policy_network_loss~��

A2S/value_network_loss�7C

A2S/q_network_loss��CV��B+       ��K	f\WHY��A�j*

A2S/average_reward_1��!B`�+       ��K	r��HY��A�m*

A2S/average_reward_1��IC��-�+       ��K	��JY��A�t*

A2S/average_reward_1�7 D6�mAw       洦�	S�RY��A�t*i

A2S/kl�;

A2S/policy_network_loss���

A2S/value_network_loss��nC

A2S/q_network_loss}�bC�P�+       ��K	�;�RY��A�u*

A2S/average_reward_1���AQi�+       ��K	���RY��A�v*

A2S/average_reward_1�]�BDD#+       ��K	P�5TY��A�~*

A2S/average_reward_1*= D���w       洦�	Pw\Y��A�~*i

A2S/kl,�1;

A2S/policy_network_loss.��

A2S/value_network_loss��iC

A2S/q_network_lossKeC���,       ���E	�#�\Y��A��*

A2S/average_reward_1�C�M,       ���E	�/�\Y��AÀ*

A2S/average_reward_1b��Aʭ�!,       ���E	���\Y��A�*

A2S/average_reward_1oU�A��1,       ���E	�]Y��A��*

A2S/average_reward_1��B�xh,       ���E	�cv^Y��A��*

A2S/average_reward_1�#%D ��Ix       ��!�	v��fY��A��*i

A2S/kl&�F;

A2S/policy_network_lossZP�

A2S/value_network_lossH PC

A2S/q_network_lossJ�KCx��I,       ���E	ay�fY��A��*

A2S/average_reward_1�n�A�3��,       ���E	s�fY��Aʉ*

A2S/average_reward_1���A�dc�,       ���E	�0�gY��A��*

A2S/average_reward_1ã�C�h�2,       ���E	��gY��A��*

A2S/average_reward_1�*JB�@Q�,       ���E	{��gY��A�*

A2S/average_reward_1[�	C���,,       ���E	�whY��AЏ*

A2S/average_reward_1r�BG��],       ���E	��iY��A��*

A2S/average_reward_13�D�zL@x       ��!�	�UqY��A��*i

A2S/kl4��<

A2S/policy_network_loss�~��

A2S/value_network_loss��&C

A2S/q_network_loss�C�|R,       ���E	��dqY��A��*

A2S/average_reward_1�vB4�[J,       ���E	^mqY��A��*

A2S/average_reward_1粰Au���,       ���E	�sY��A��*

A2S/average_reward_1��"Dmc%{x       ��!�	Y�@{Y��A��*i

A2S/kl���:

A2S/policy_network_lossNU	�

A2S/value_network_lossx��C

A2S/q_network_loss ��CH�E,       ���E	�,P{Y��A͞*

A2S/average_reward_1�n/B�dNk,       ���E	Uo|Y��Aʢ*

A2S/average_reward_1{�CA�7f,       ���E	Sq8|Y��Aϣ*

A2S/average_reward_1�e�B���,       ���E	�A|Y��A��*

A2S/average_reward_1�#B�T,       ���E	�:�|Y��A��*

A2S/average_reward_1۩�C����x       ��!�	�9�Y��A��*i

A2S/klFw8;

A2S/policy_network_loss�¿

A2S/value_network_loss��C

A2S/q_network_loss7�Bn�,       ���E	�%d�Y��Aߩ*

A2S/average_reward_13�ZC-�o/,       ���E	YC��Y��AǱ*

A2S/average_reward_1J"D�~�,       ���E	����Y��A��*

A2S/average_reward_1NB����x       ��!�	�R�Y��A��*i

A2S/klRNT:

A2S/policy_network_loss%�

A2S/value_network_lossRYgC

A2S/q_network_lossҩcC."�.,       ���E	u��Y��A��*

A2S/average_reward_1U�A�FX7,       ���E	)M؏Y��A��*

A2S/average_reward_1&�C���?,       ���E	6yl�Y��A��*

A2S/average_reward_1�ΚC~�#Mx       ��!�	����Y��A��*i

A2S/klx��:

A2S/policy_network_loss���

A2S/value_network_loss�`qC

A2S/q_network_lossynC$y\6,       ���E	����Y��A˺*

A2S/average_reward_1i�B[��I,       ���E	�DǘY��A��*

A2S/average_reward_1Y�%B�۹,       ���E	���Y��AԻ*

A2S/average_reward_1a��A��R|,       ���E	���Y��A�*

A2S/average_reward_1
l�B�]=�,       ���E	�羙Y��A��*

A2S/average_reward_1@�C~��#,       ���E	�k�Y��A��*

A2S/average_reward_1�BJ� ,       ���E	�S�Y��A��*

A2S/average_reward_1�<>C��3,       ���E	MA��Y��A��*

A2S/average_reward_1��`C)�5,       ���E	�b=�Y��A��*

A2S/average_reward_1�wCv��x       ��!�	�HK�Y��A��*i

A2S/kl�};

A2S/policy_network_loss5Sw�

A2S/value_network_lossꌶB

A2S/q_network_loss�[�B8���,       ���E	��Y��A��*

A2S/average_reward_1l�C��F,       ���E	�'�Y��A��*

A2S/average_reward_1¢SB�$�,       ���E	荥Y��A��*

A2S/average_reward_1j%D�ەw,       ���E	�z�Y��A��*

A2S/average_reward_17�#D�p:x       ��!�	�nj�Y��A��*i

A2S/kl��;

A2S/policy_network_lossD�y�

A2S/value_network_lossDKOC

A2S/q_network_lossO�HC�,�,       ���E	/qٯY��A��*

A2S/average_reward_1��9C��l�,       ���E	�7/�Y��A��*

A2S/average_reward_1-�D���,       ���E	�kK�Y��A��*

A2S/average_reward_1z'0B�@ x       ��!�	�Y��Y��A��*i

A2S/kl�j;

A2S/policy_network_loss����

A2S/value_network_loss)[C

A2S/q_network_loss��VC�O�W,       ���E	}ߺY��A��*

A2S/average_reward_1-�&D��,       ���E	3���Y��A��*

A2S/average_reward_1 	Bu9#,       ���E	�Kp�Y��A��*

A2S/average_reward_1�g(D��Q�x       ��!�	�b��Y��A��*i

A2S/kl�;

A2S/policy_network_loss~J�

A2S/value_network_lossJ�tC

A2S/q_network_lossZ�hC�ٰ�,       ���E	�� �Y��A��*

A2S/average_reward_1�
B��=�,       ���E	Z�Y��A��*

A2S/average_reward_1�+0B���,       ���E	�Mq�Y��A��*

A2S/average_reward_1-�*D�٠cx       ��!�	%���Y��A��*i

A2S/kl�-;

A2S/policy_network_loss���

A2S/value_network_loss�oC

A2S/q_network_loss�1pCG��,       ���E	%���Y��A��*

A2S/average_reward_1���A�l�1,       ���E	y�[�Y��A߈*

A2S/average_reward_1;+D^x       ��!�	w=}�Y��A߈*i

A2S/klE�/;

A2S/policy_network_loss>���

A2S/value_network_loss%|�C

A2S/q_network_loss\J�Cj�^,       ���E	����Y��A��*

A2S/average_reward_1�-A��,       ���E	�ä�Y��AÉ*

A2S/average_reward_1�>B�ٚ,       ���E	N��Y��A�*

A2S/average_reward_1�k�A`\H5,       ���E	g��Y��Aˋ*

A2S/average_reward_1I0CE;�,       ���E	s��Y��A��*

A2S/average_reward_1�BQ�,       ���E	��'�Y��A��*

A2S/average_reward_1j(B��n�,       ���E	��0�Y��Aь*

A2S/average_reward_1�`�AO\�&,       ���E	��<�Y��A��*

A2S/average_reward_1J�B� .,       ���E	��H�Y��A��*

A2S/average_reward_1��cA��u,       ���E	�ϗ�Y��A�*

A2S/average_reward_1��CyƳ�,       ���E	����Y��A��*

A2S/average_reward_1�#C9�d�,       ���E	��X�Y��A��*

A2S/average_reward_1'D��x       ��!�	�Ŕ�Y��A��*i

A2S/kl��<<

A2S/policy_network_loss[V˿

A2S/value_network_loss>~UC

A2S/q_network_lossm QC�)t,       ���E	����Y��A��*

A2S/average_reward_1bD<��p,       ���E	��%�Y��A�*

A2S/average_reward_1�+D�gX,       ���E	�`�Y��Aˮ*

A2S/average_reward_1��+D��?�x       ��!�	����Y��Aˮ*i

A2S/kl~Cp:

A2S/policy_network_lossO��

A2S/value_network_lossApaC

A2S/q_network_lossh�[C��!,       ���E	���Y��A�*

A2S/average_reward_1l�AS?��,       ���E	�
��Y��A��*

A2S/average_reward_1��AX;&-,       ���E	e�Y��A�*

A2S/average_reward_1�n.D�A@x       ��!�	//,�Y��A�*i

A2S/klz�;

A2S/policy_network_loss[5D�

A2S/value_network_loss�ɐC

A2S/q_network_loss��C[8�,       ���E	��=�Y��A��*

A2S/average_reward_17~�A]%,       ���E	۾5�Y��A��*

A2S/average_reward_1U��C1{-x       ��!�	�$AZ��A��*i

A2S/kl��;

A2S/policy_network_loss�/o�

A2S/value_network_loss�LC

A2S/q_network_lossZ�PCk-Q!,       ���E	J��Z��A��*

A2S/average_reward_1dӚC#���,       ���E	X��Z��Aǿ*

A2S/average_reward_1e��A�S,       ���E	�;IZ��A��*

A2S/average_reward_1�f2D��v�x       ��!�	C�Z��A��*i

A2S/kl={�:

A2S/policy_network_loss`]��

A2S/value_network_loss�ՐC

A2S/q_network_loss:��C��w,       ���E	;��Z��A��*

A2S/average_reward_1��A=t��,       ���E	��Z��A��*

A2S/average_reward_1��B�� �,       ���E	��Z��A��*

A2S/average_reward_1��A��{,       ���E	��Z��A��*

A2S/average_reward_1<~�A�?�,       ���E	<�dZ��A��*

A2S/average_reward_1�J�CU��,       ���E	D�qZ��A��*

A2S/average_reward_1ǞAO���,       ���E	�1zZ��A��*

A2S/average_reward_1�C�A�Cs�,       ���E	��Z��A��*

A2S/average_reward_1��Ab���,       ���E	AD�Z��A��*

A2S/average_reward_1��aC)�},       ���E	
a�Z��A��*

A2S/average_reward_1_�+A]��1,       ���E	��Z��A��*

A2S/average_reward_1�~�Aez��,       ���E	�_Z��A��*

A2S/average_reward_1��]C;e��,       ���E	�Z��A��*

A2S/average_reward_1F�+D�Y��,       ���E	��Z��A��*

A2S/average_reward_1�D��?,       ���E	z5�Z��A��*

A2S/average_reward_1�A��&�x       ��!�	�Z��A��*i

A2S/kl1�<

A2S/policy_network_loss����

A2S/value_network_loss�aC

A2S/q_network_loss�KdC�>��,       ���E	b2`Z��A��*

A2S/average_reward_1d0D�b�*,       ���E	�mZ��A��*

A2S/average_reward_1#�B"x�%,       ���E	�$uZ��A��*

A2S/average_reward_1�+�Az$Ⱦ,       ���E	��~Z��A��*

A2S/average_reward_1{mA���,       ���E	Y��Z��A��*

A2S/average_reward_1B".D�,       ���E	�1�Z��A��*

A2S/average_reward_1��LB���,       ���E	z>XZ��A��*

A2S/average_reward_1�r-D6^:Z,       ���E	E˧Z��A�*

A2S/average_reward_1c.D��g�,       ���E	�!�Z��A��*

A2S/average_reward_1���A�,:x       ��!�	mC'Z��A��*i

A2S/kl g :

A2S/policy_network_lossݖD�

A2S/value_network_lossb|C

A2S/q_network_lossnxC˗�`,       ���E	�4p(Z��Aʈ*

A2S/average_reward_1pwD��|�,       ���E	��w(Z��Aވ*

A2S/average_reward_1\��A�?�,       ���E	��(Z��A��*

A2S/average_reward_1A>�A�ʾ,       ���E	�a�)Z��A�*

A2S/average_reward_1y3D��pA,       ���E	G>�)Z��A��*

A2S/average_reward_10s�A]���x       ��!�	iL2Z��A��*i

A2S/kl�5:

A2S/policy_network_loss����

A2S/value_network_loss�lC

A2S/q_network_loss�WiC��w,       ���E	�<"3Z��A��*

A2S/average_reward_1�/�Cծ ,       ���E	=�/3Z��A�*

A2S/average_reward_1a��A�U��,       ���E	8!93Z��A��*

A2S/average_reward_1W3 B�G�,       ���E	4E3Z��A��*

A2S/average_reward_1pB�(x,       ���E	� �4Z��A��*

A2S/average_reward_1@�/D���,       ���E	���5Z��A��*

A2S/average_reward_1�+Dti�,       ���E	|2 6Z��A��*

A2S/average_reward_1���A3};},       ���E	n@�6Z��A��*

A2S/average_reward_1�Y�Ck]�x,       ���E	ۭ�6Z��A��*

A2S/average_reward_1>2B���,       ���E	K��6Z��A�*

A2S/average_reward_1}�&B�kK,       ���E	u)97Z��A��*

A2S/average_reward_1�HCf4D�,       ���E	A�A7Z��A˭*

A2S/average_reward_1�J
BL!��,       ���E	�U7Z��A��*

A2S/average_reward_1F>B�c��,       ���E	���7Z��A��*

A2S/average_reward_1�	C��,       ���E	��$9Z��A�*

A2S/average_reward_1�l+D��,       ���E	+�49Z��A��*

A2S/average_reward_1�t�Av���,       ���E	��=9Z��A��*

A2S/average_reward_1���A�w�O,       ���E	وK9Z��A�*

A2S/average_reward_1��cA+�Cx       ��!�	I}�AZ��A�*i

A2S/kl��+;

A2S/policy_network_loss;ZH�

A2S/value_network_lossd�C

A2S/q_network_loss��C�Ę�,       ���E	�q�AZ��A��*

A2S/average_reward_1�a'B5/��,       ���E	�;�AZ��Aӹ*

A2S/average_reward_1TS�A���,       ���E	RY�AZ��A��*

A2S/average_reward_1�CbB=:��,       ���E	;�AZ��AǺ*

A2S/average_reward_1�K�AEM��,       ���E	�0�AZ��Aں*

A2S/average_reward_1���A��J�,       ���E	Q�EBZ��A��*

A2S/average_reward_1�C��,       ���E	��NBZ��A��*

A2S/average_reward_11��AM�j1,       ���E	�VBZ��Aм*

A2S/average_reward_1�A�+,       ���E	�"�BZ��A��*

A2S/average_reward_1rh�C���,       ���E	�zDZ��A��*

A2S/average_reward_1�a,DA�Ď,       ���E	-�iEZ��A��*

A2S/average_reward_1�D��X�,       ���E	�ZqEZ��A��*

A2S/average_reward_1@��A�"��,       ���E	Os�EZ��A��*

A2S/average_reward_1���B	rj�,       ���E	��EZ��A��*

A2S/average_reward_1�FBE�,       ���E	���EZ��A��*

A2S/average_reward_1���B+[�,       ���E	�3GZ��A��*

A2S/average_reward_1��'D��{+,       ���E	3CGZ��A��*

A2S/average_reward_1�y B|�,W,       ���E	���GZ��A��*

A2S/average_reward_1��B^�,       ���E	r�GZ��A��*

A2S/average_reward_1�{)B��,       ���E	�IZ��A��*

A2S/average_reward_1V�&Dk�!
,       ���E	�OIZ��A��*

A2S/average_reward_1z�Be��,       ���E	�4IZ��A��*

A2S/average_reward_1e�AJ=�,       ���E	�?IZ��A��*

A2S/average_reward_1}i�Ar,�,       ���E	qq�IZ��A��*

A2S/average_reward_1�P�C�Y��,       ���E	�L�IZ��A��*

A2S/average_reward_1�gB�<�,,       ���E	~PPJZ��A��*

A2S/average_reward_1�YC�,       ���E	��TJZ��A��*

A2S/average_reward_1���A��s�,       ���E	�]^JZ��A��*

A2S/average_reward_13��A��%�,       ���E	��KZ��A��*

A2S/average_reward_1��-D��,       ���E	ϑ?LZ��A��*

A2S/average_reward_1�5�C����,       ���E	��MLZ��A��*

A2S/average_reward_1\��A��,       ���E	p7bLZ��A��*

A2S/average_reward_1HwBR�a,       ���E	�hLZ��A��*

A2S/average_reward_1��Akܹ�,       ���E	~�MZ��A��*

A2S/average_reward_1�-D��Ť,       ���E	g��NZ��A��*

A2S/average_reward_1��DI��,       ���E	�5PZ��A�*

A2S/average_reward_1�,D@Xl�,       ���E	a\'PZ��A��*

A2S/average_reward_1�V�AYi��,       ���E	3��PZ��A�*

A2S/average_reward_1OtC>�e,       ���E	"��PZ��A��*

A2S/average_reward_1EG�Ay-��,       ���E	�rQZ��A�*

A2S/average_reward_1��CԘU,       ���E	���QZ��A��*

A2S/average_reward_1���A��K&,       ���E	#��QZ��AΓ*

A2S/average_reward_1��A6�X,       ���E	c��QZ��A��*

A2S/average_reward_1p�'B,Й�,       ���E	T�QZ��A��*

A2S/average_reward_1r�B�P�3,       ���E	���QZ��AӔ*

A2S/average_reward_1�\B���m,       ���E	0f0RZ��A��*

A2S/average_reward_1ps<C)b+,       ���E	��sSZ��A��*

A2S/average_reward_1�-Dv�d�x       ��!�	7��[Z��A��*i

A2S/klip#9

A2S/policy_network_loss`p�

A2S/value_network_lossE�C

A2S/q_network_loss�p�C�B�,       ���E	u�[Z��A��*

A2S/average_reward_1�(B�Y�,       ���E	��\Z��A��*

A2S/average_reward_1��AL�,       ���E	ix\Z��AП*

A2S/average_reward_12��A �9�,       ���E	4,\Z��A�*

A2S/average_reward_1�t�A�,��,       ���E	�'\Z��A��*

A2S/average_reward_1Ue�AbT,       ���E	=�\Z��AȢ*

A2S/average_reward_1m�mC�c�",       ���E	���\Z��A��*

A2S/average_reward_1$�8B�=�!,       ���E	y��\Z��A֣*

A2S/average_reward_1e� B�.�L,       ���E	���\Z��A�*

A2S/average_reward_1�g�B%��,       ���E	�c]Z��A��*

A2S/average_reward_1��A�J',       ���E	,�]Z��A��*

A2S/average_reward_1�K�A<�f�,       ���E	?_�^Z��A��*

A2S/average_reward_1�,Dd/��,       ���E	���_Z��A��*

A2S/average_reward_1(+D�K�	,       ���E	Q(�_Z��A��*

A2S/average_reward_1�*�A�>h�,       ���E	��e`Z��A��*

A2S/average_reward_1mvCc`&,       ���E	�s�aZ��A��*

A2S/average_reward_1Z�.D,f�",       ���E	���bZ��A��*

A2S/average_reward_1l�,D�u�,       ���E	�<QdZ��A��*

A2S/average_reward_1�;+D�%,       ���E	��dZ��A��*

A2S/average_reward_1�f�C��,       ���E	�geZ��A��*

A2S/average_reward_1�8B"��,       ���E	��eZ��A��*

A2S/average_reward_1"�A[:e,       ���E	0�eZ��A��*

A2S/average_reward_1�j�C����,       ���E	ڬ�eZ��A��*

A2S/average_reward_1�EB�au3x       ��!�	�T�nZ��A��*i

A2S/kl-�<

A2S/policy_network_lossJ��

A2S/value_network_loss�MqC

A2S/q_network_loss��gC��Z�,       ���E	���nZ��A��*

A2S/average_reward_1i�B�/N>,       ���E	��EoZ��A��*

A2S/average_reward_1��Cʈ�r,       ���E	�IKoZ��A��*

A2S/average_reward_1��ASU��,       ���E	�RoZ��A��*

A2S/average_reward_1˓�A�|8�,       ���E	r\voZ��A��*

A2S/average_reward_1��BM-+�,       ���E	N݃oZ��A��*

A2S/average_reward_1K\(B�~6I,       ���E	z�oZ��A��*

A2S/average_reward_1�A B���l,       ���E	_ޚoZ��A��*

A2S/average_reward_1�'�AT+Ԣ,       ���E	_kpZ��A��*

A2S/average_reward_1[ޙC3�7,       ���E	�pZ��A��*

A2S/average_reward_1!�A�y�,       ���E	��upZ��A��*

A2S/average_reward_1�J*C�+�,       ���E	g�{pZ��A��*

A2S/average_reward_1�|�A��,       ���E	(ÒpZ��A��*

A2S/average_reward_100�Aq9�l,       ���E	NI�pZ��A��*

A2S/average_reward_1u��A%-�,       ���E	OfqZ��A��*

A2S/average_reward_1�ЦC���,       ���E	�QwrZ��A��*

A2S/average_reward_1f�D����,       ���E	�A�rZ��A��*

A2S/average_reward_1?Y�A펢X,       ���E	ΎrZ��A��*

A2S/average_reward_1��B¼�E,       ���E	�ҢrZ��A��*

A2S/average_reward_1�SBk쭷,       ���E	t��rZ��A��*

A2S/average_reward_1�T&B�-��,       ���E	���sZ��A��*

A2S/average_reward_1a�D�<l�,       ���E	H��sZ��A��*

A2S/average_reward_1���A��ˎ,       ���E	�
KuZ��A��*

A2S/average_reward_1/(D7���,       ���E	��uZ��A��*

A2S/average_reward_1�C4N&�,       ���E	��vZ��A�*

A2S/average_reward_1b�C��Ө,       ���E	N�vZ��A��*

A2S/average_reward_1faB����,       ���E	��vZ��A��*

A2S/average_reward_1�^%AoS��,       ���E	��:xZ��A��*

A2S/average_reward_1�+D�q��,       ���E	�1�xZ��A��*

A2S/average_reward_1@,&C��,       ���E	�2�xZ��A��*

A2S/average_reward_1�FC��9,       ���E	��yZ��A͒*

A2S/average_reward_1��CC����,       ���E	��zZ��A��*

A2S/average_reward_1,.D�Kdz,       ���E	���zZ��A�*

A2S/average_reward_1ܪ�A�(LF,       ���E	�W�{Z��Aʢ*

A2S/average_reward_1�,Dv�y�,       ���E	9y�{Z��A��*

A2S/average_reward_1@=FB�)�,       ���E	��|Z��A��*

A2S/average_reward_1���A��,       ���E	��(|Z��A��*

A2S/average_reward_1WTB�e�:,       ���E	�g;|Z��A��*

A2S/average_reward_1��B�U�,       ���E	b;�}Z��A��*

A2S/average_reward_1X{*D��(x       ��!�	 u��Z��A��*i

A2S/kl9^�<

A2S/policy_network_loss�~��

A2S/value_network_lossZ`qC

A2S/q_network_loss:qCJH�,       ���E	杆Z��A��*

A2S/average_reward_1���A�`�,       ���E	���Z��A�*

A2S/average_reward_1GjC�y��,       ���E	��Z��Aî*

A2S/average_reward_1
�AB���],       ���E	���Z��Aݮ*

A2S/average_reward_1�N	A)�I�,       ���E	�v9�Z��A��*

A2S/average_reward_1d�B��ϩ,       ���E	J�L�Z��A�*

A2S/average_reward_1�nBl8n�,       ���E	�WS�Z��A��*

A2S/average_reward_1���A��*@,       ���E	���Z��A��*

A2S/average_reward_19gC^e),       ���E	���Z��A�*

A2S/average_reward_1ݙC�7�H,       ���E	��Z��A��*

A2S/average_reward_1��A���,       ���E	b�Z��Aγ*

A2S/average_reward_1��Av w,       ���E	��8�Z��A�*

A2S/average_reward_1	 �B�:,       ���E	��?�Z��A��*

A2S/average_reward_1�+�A���,       ���E	��Q�Z��A��*

A2S/average_reward_1��A<�nH,       ���E	���Z��A��*

A2S/average_reward_1nA�C��!�,       ���E	Ta�Z��A��*

A2S/average_reward_1��"D���,       ���E	2%��Z��A��*

A2S/average_reward_1<~-D"��,       ���E	��ɋZ��A��*

A2S/average_reward_1�5B�`,       ���E	
��Z��A��*

A2S/average_reward_1��(D�G��,       ���E	2h��Z��A��*

A2S/average_reward_1pݨC_嫾,       ���E	���Z��A��*

A2S/average_reward_1�t�A�l>,       ���E	NRʍZ��A��*

A2S/average_reward_1�y�A���,       ���E	�NÎZ��A��*

A2S/average_reward_1���C��~,       ���E	�JˎZ��A��*

A2S/average_reward_1�S�A�=ZG,       ���E	/�3�Z��A��*

A2S/average_reward_1�Q)D��3�,       ���E	�Gh�Z��A��*

A2S/average_reward_1��Ct'�u,       ���E	r$��Z��A��*

A2S/average_reward_1�9,DYx�:,       ���E	�ǑZ��A��*

A2S/average_reward_1�8�A�EN�,       ���E	�Z��A��*

A2S/average_reward_1F�D���Ix       ��!�	I<�Z��A��*i

A2S/kleL�:

A2S/policy_network_loss!BZ�

A2S/value_network_loss��ZC

A2S/q_network_loss�XC�=Z,       ���E	�81�Z��A��*

A2S/average_reward_1��A�W��,       ���E	A�:�Z��A��*

A2S/average_reward_1�/�A��,       ���E	R2I�Z��A��*

A2S/average_reward_1"ٵA��y,       ���E	����Z��A��*

A2S/average_reward_1�,D3;��,       ���E	Z���Z��A�*

A2S/average_reward_1�-DQ쩈,       ���E	|��Z��A��*

A2S/average_reward_1{�A�D�,       ���E	��Z��A��*

A2S/average_reward_1���A�_?�,       ���E	� d�Z��A��*

A2S/average_reward_1��.D�~k,       ���E	VFr�Z��Aċ*

A2S/average_reward_1�B�(,       ���E	<��Z��A��*

A2S/average_reward_1{�A[F�w,       ���E	�ː�Z��A��*

A2S/average_reward_1��=B!,/�,       ���E	�=�Z��Aߐ*

A2S/average_reward_11��C�o,       ���E	KVF�Z��A��*

A2S/average_reward_1Jo�A�#k�,       ���E	�6s�Z��A��*

A2S/average_reward_1k��B*�%,       ���E	އ��Z��A��*

A2S/average_reward_1�E�A�e�vx       ��!�	~�Z��A��*i

A2S/kl��m:

A2S/policy_network_loss���

A2S/value_network_lossG�C

A2S/q_network_loss1��C��W6,       ���E	vc�Z��A��*

A2S/average_reward_1�0Dι�6,       ���E	��o�Z��AΚ*

A2S/average_reward_1���AL)E�,       ���E	'��Z��Aמ*

A2S/average_reward_1�B�CG^�X,       ���E	'�J�Z��A��*

A2S/average_reward_1�,�B�>�V,       ���E	QG��Z��A£*

A2S/average_reward_12�CcU~�,       ���E	���Z��A��*

A2S/average_reward_1�DPVkj,       ���E	q8�Z��A��*

A2S/average_reward_1���C���\,       ���E	A��Z��A��*

A2S/average_reward_1y+�A탍�,       ���E	�bٯZ��A��*

A2S/average_reward_1�!�C�
[,       ���E	���Z��Aȴ*

A2S/average_reward_1>��A�~�,       ���E	C���Z��A��*

A2S/average_reward_1�`�Bu|�,       ���E	jv�Z��A��*

A2S/average_reward_1�/�Cz F,       ���E	J��Z��Aظ*

A2S/average_reward_1V/BoFa%,       ���E	!З�Z��A��*

A2S/average_reward_1P�"B\ ��,       ���E	8�αZ��A��*

A2S/average_reward_1��3D^Q]x       ��!�	gt��Z��A��*i

A2S/kl�6;

A2S/policy_network_loss@�ƿ

A2S/value_network_loss`flC

A2S/q_network_loss��fC�(]�,       ���E	���Z��A��*

A2S/average_reward_1�n�A��S�,       ���E	��R�Z��A��*

A2S/average_reward_1R�Cڛ],       ���E	4���Z��A��*

A2S/average_reward_1�4D5Հ,       ���E	��ǼZ��A��*

A2S/average_reward_1��EBd7�P,       ���E	�мZ��A��*

A2S/average_reward_1���A
�Ѝ,       ���E	�E׼Z��A��*

A2S/average_reward_1Jk�A�S��,       ���E	�$�Z��A��*

A2S/average_reward_1Ë�B+0v{,       ���E	"f�Z��A��*

A2S/average_reward_1{�A)�,       ���E	|��Z��A��*

A2S/average_reward_1���A�Dc,       ���E	?�)�Z��A��*

A2S/average_reward_1�"�A��w,,       ���E	��2�Z��A��*

A2S/average_reward_1�>�ASx��,       ���E	ZwE�Z��A��*

A2S/average_reward_1wBl$�>,       ���E	L)X�Z��A��*

A2S/average_reward_1c}2B���,       ���E	f���Z��A��*

A2S/average_reward_1l�3D`�h,       ���E	,��Z��A��*

A2S/average_reward_1{��A��s�,       ���E	Ce;Z��A��*

A2S/average_reward_1�BS�,
,       ���E	�M־Z��A��*

A2S/average_reward_1C��A���,       ���E	Ф��Z��A��*

A2S/average_reward_1�/�C���",       ���E	ދC�Z��A��*

A2S/average_reward_1�OxC�՞�,       ���E	��M�Z��A��*

A2S/average_reward_1���A���,       ���E	���Z��A��*

A2S/average_reward_1Mf)Dvf�m,       ���E	����Z��A��*

A2S/average_reward_1�#B���,       ���E	�}��Z��A��*

A2S/average_reward_1���BVLF�,       ���E	����Z��A��*

A2S/average_reward_1���Axy�,       ���E	���Z��A��*

A2S/average_reward_1۞�Aw�S�,       ���E	�d]�Z��A��*

A2S/average_reward_1��*D���,       ���E	��A�Z��A��*

A2S/average_reward_1���CQUƮ,       ���E	&���Z��A��*

A2S/average_reward_1n�-DO��=,       ���E	���Z��A��*

A2S/average_reward_1'>!C��`r,       ���E	���Z��A��*

A2S/average_reward_1�
�Az�<o,       ���E	� �Z��A܄*

A2S/average_reward_1�?�C'=��x       ��!�	�))�Z��A܄*i

A2S/kl6��;

A2S/policy_network_loss&��

A2S/value_network_loss�SC

A2S/q_network_loss�_RC݉�,       ���E	�@�Z��A��*

A2S/average_reward_1��A�:�:,       ���E	l�q�Z��A��*

A2S/average_reward_1Ǘ�B�'r,       ���E	Ҥ��Z��AȆ*

A2S/average_reward_1��9B�|,       ���E	�Z��Z��A��*

A2S/average_reward_1S�A�C׷,       ���E	Z�D�Z��A��*

A2S/average_reward_1f��C����,       ���E	��L�Z��A׊*

A2S/average_reward_1x#�A<��V,       ���E	�;Y�Z��A��*

A2S/average_reward_1_w&B4�3W,       ���E	)�f�Z��A��*

A2S/average_reward_1���A���,       ���E	;�r�Z��A��*

A2S/average_reward_1��Ato<�,       ���E	�|��Z��A��*

A2S/average_reward_1��KB��+,       ���E	���Z��AҌ*

A2S/average_reward_1��A��Y,       ���E	����Z��A��*

A2S/average_reward_1K��AKQB3,       ���E	�|+�Z��A�*

A2S/average_reward_1-5D&�,       ���E	V5�Z��A��*

A2S/average_reward_1y�AA��`,       ���E	�Z�Z��A�*

A2S/average_reward_1���B�� B,       ���E	_/��Z��Aח*

A2S/average_reward_1g�4C�[��x       ��!�	+Q'�Z��Aח*i

A2S/kl��K;

A2S/policy_network_loss�n}�

A2S/value_network_loss9��C

A2S/q_network_loss-�C���7,       ���E	6�r�Z��A��*

A2S/average_reward_1�$3D�s�,       ���E	��z�Z��A۟*

A2S/average_reward_1M��A܆5�,       ���E	bۻ�Z��Aç*

A2S/average_reward_1��5D�A%�,       ���E	G��Z��A��*

A2S/average_reward_1a�3D�Hfj,       ���E	�,�Z��A��*

A2S/average_reward_1��B�an,       ���E	�{��Z��A�*

A2S/average_reward_1�O3D�=��,       ���E	*���Z��A��*

A2S/average_reward_1@B�Y=.,       ���E	�U��Z��A��*

A2S/average_reward_1��A0��D,       ���E	 ��Z��A��*

A2S/average_reward_1�hC�E�C,       ���E	B�@�Z��Aֻ*

A2S/average_reward_1�pC8�V,       ���E	��J�Z��A��*

A2S/average_reward_1��ATI�x,       ���E	�>��Z��A��*

A2S/average_reward_1��5D�LT8,       ���E	�>��Z��A��*

A2S/average_reward_15ID��j�,       ���E	�y9�Z��A��*

A2S/average_reward_1q�2D���@,       ���E	��@�Z��A��*

A2S/average_reward_1-1�A��k ,       ���E	8��Z��A��*

A2S/average_reward_1<V2Dz��x       ��!�	��Z��A��*i

A2S/klǄ?;

A2S/policy_network_lossF���

A2S/value_network_loss+O�C

A2S/q_network_loss���C�;s�,       ���E	Q��Z��A��*

A2S/average_reward_1��)B.��K,       ���E	+m_�Z��A��*

A2S/average_reward_1dv6D�{�,       ���E	ho��Z��A��*

A2S/average_reward_1;iCg-�1,       ���E	T4��Z��A��*

A2S/average_reward_1BÙ��,       ���E	�Q��Z��A��*

A2S/average_reward_1�B�B�1,       ���E	���Z��A��*

A2S/average_reward_1�&�B��v,       ���E	�c"�Z��A��*

A2S/average_reward_1��SB�<V�,       ���E	�,�Z��A��*

A2S/average_reward_1�RB>�T�,       ���E	`�3�Z��A��*

A2S/average_reward_1D�A�',       ���E	���Z��A��*

A2S/average_reward_1�pCI3��,       ���E	ȝ�Z��A��*

A2S/average_reward_1'5�A�V��,       ���E	����Z��A��*

A2S/average_reward_1k�Ai��,       ���E	Kh��Z��A��*

A2S/average_reward_1CW�ADd54,       ���E	]E��Z��A��*

A2S/average_reward_14A�A��,       ���E	Ѡ.�Z��A��*

A2S/average_reward_1V�}COsf,       ���E	�l7�Z��A��*

A2S/average_reward_1��B�.�x       ��!�	�Ҳ�Z��A��*i

A2S/kl��^;

A2S/policy_network_loss�J�

A2S/value_network_loss�ӁC

A2S/q_network_loss�l�C	��e,       ���E	�X��Z��A��*

A2S/average_reward_1���A�aɈ,       ���E	Y���Z��A��*

A2S/average_reward_1�xB���,       ���E	�x��Z��A��*

A2S/average_reward_1�
�C	�9,       ���E	����Z��A��*

A2S/average_reward_1	��Ae�N,       ���E	����Z��A��*

A2S/average_reward_1h��B��SR,       ���E	��Z��A��*

A2S/average_reward_1~��A턁�,       ���E	BP
�Z��A��*

A2S/average_reward_1e��AM1M�,       ���E	+`-�Z��A��*

A2S/average_reward_1fi�Bo��W,       ���E	L8�Z��A��*

A2S/average_reward_1]��A4 2,       ���E	��W�Z��A��*

A2S/average_reward_1.�BaH�,       ���E	:�_�Z��A��*

A2S/average_reward_1�ɟA$A{�,       ���E	�m�Z��A��*

A2S/average_reward_1��A���,       ���E	^fz�Z��A��*

A2S/average_reward_1� �AĜ��,       ���E	���Z��A��*

A2S/average_reward_1W��A�\��,       ���E	-��Z��A��*

A2S/average_reward_1��B]�f�,       ���E	"��Z��A��*

A2S/average_reward_1��AL�,u,       ���E	����Z��A��*

A2S/average_reward_1���B�ܻ�,       ���E	3�%�Z��A��*

A2S/average_reward_1�
C;�b,       ���E	����Z��A��*

A2S/average_reward_1m8�Cۥ�,       ���E	�z��Z��A��*

A2S/average_reward_1�y�A�=��,       ���E	��[��A��*

A2S/average_reward_1=YDOEi�,       ���E	�&[��Aƃ*

A2S/average_reward_1-�@v�͢,       ���E	���[��A��*

A2S/average_reward_1H�wC��X,       ���E	��[��Aʆ*

A2S/average_reward_1A�B�0��,       ���E	17�[��A�*

A2S/average_reward_1���A�p,       ���E	J�[��A��*

A2S/average_reward_1mHC�Խ�,       ���E	�m[��A��*

A2S/average_reward_1�*D8��,       ���E	oOy[��Aǐ*

A2S/average_reward_1Ae�@��M�,       ���E	6:�[��A��*

A2S/average_reward_1��%B���,       ���E	��[��A��*

A2S/average_reward_1ܷA��1,       ���E	��K[��A��*

A2S/average_reward_1��C �,       ���E	�~V[��Aӕ*

A2S/average_reward_1u��A�E�,       ���E	{:�[��A��*

A2S/average_reward_1��,DLkk#,       ���E	8[��A��*

A2S/average_reward_1��*Dh��,       ���E	��&[��A�*

A2S/average_reward_1,8BIQ�,       ���E	�[��A��*

A2S/average_reward_1�4�C���L,       ���E	l�>	[��A�*

A2S/average_reward_1�+D�?$a,       ���E	`�N	[��A��*

A2S/average_reward_1�PB���<,       ���E	�^[	[��Aϲ*

A2S/average_reward_1��"B���f,       ���E	�h	[��A��*

A2S/average_reward_1��+BS�C�,       ���E	�t	[��A��*

A2S/average_reward_1�(�A���!,       ���E	CV�	[��A�*

A2S/average_reward_1I%fC=���,       ���E	��
[��A��*

A2S/average_reward_1��aB�FL,       ���E	�r[��A��*

A2S/average_reward_1u�,D�8�^,       ���E	�m�[��A۾*

A2S/average_reward_1��?B�n�,       ���E	��[��A��*

A2S/average_reward_1���A�P��,       ���E	��[��A��*

A2S/average_reward_1�i�Cq]�,       ���E	�\"[��A��*

A2S/average_reward_1��-B.��,       ���E	�B[��A��*

A2S/average_reward_1b��B7�,       ���E	u[��A��*

A2S/average_reward_1���B�|�,       ���E	��[��A��*

A2S/average_reward_1��B��Mm,       ���E	G6[��A��*

A2S/average_reward_1��C.��L,       ���E	�$[��A��*

A2S/average_reward_1	"BA	�,       ���E	U3[��A��*

A2S/average_reward_1�kB��=�,       ���E	�@�[��A��*

A2S/average_reward_1�T_C)�hm,       ���E	�[��A��*

A2S/average_reward_1>�A���`,       ���E	�!�[��A��*

A2S/average_reward_1�4�A{�@,       ���E	syB[��A��*

A2S/average_reward_1�ʜC�� u,       ���E	�(�[��A��*

A2S/average_reward_1$,D9��0,       ���E	23�[��A��*

A2S/average_reward_1�Bh
Cq,       ���E	*5�[��A��*

A2S/average_reward_1<>,DgEz,       ���E	AC�[��A��*

A2S/average_reward_1R�C?��),       ���E	��[��A��*

A2S/average_reward_1D>B���,       ���E	9/[��A��*

A2S/average_reward_1)0*D}��,       ���E	|�)[��A��*

A2S/average_reward_1c��C\��k,       ���E	�t[��A��*

A2S/average_reward_1��+D��-�,       ���E	�M�[��A��*

A2S/average_reward_1�\�A�Z�%,       ���E	1n�[��A��*

A2S/average_reward_1��@��=,       ���E	�7m[��A��*

A2S/average_reward_1�8�C�]%�,       ���E	sC�[��A��	*

A2S/average_reward_10K0Dj$V�x       ��!�	��p![��A��	*i

A2S/kl���<

A2S/policy_network_loss�}��

A2S/value_network_loss��]C

A2S/q_network_loss�`ZC+���,       ���E	�X�![��A�	*

A2S/average_reward_1�~B�%k,       ���E	Mm�![��A��	*

A2S/average_reward_1��Cn�/�,       ���E	h#[��A��	*

A2S/average_reward_1�I4DS�q,       ���E	g�-#[��Aˏ	*

A2S/average_reward_1�	B��AG,       ���E	��B#[��A��	*

A2S/average_reward_1��*B�R)�,       ���E	��O#[��A��	*

A2S/average_reward_1GfB�/�,       ���E	�K8$[��A��	*

A2S/average_reward_1Y` D�ːf,       ���E	S��$[��A��	*

A2S/average_reward_1'-OCm"��,       ���E	��G%[��A��	*

A2S/average_reward_1�AZC����,       ���E	':R%[��A��	*

A2S/average_reward_1l�A�m�,       ���E	�7]%[��A��	*

A2S/average_reward_1�B��7Z,       ���E	��}%[��A��	*

A2S/average_reward_1�qB�o@5,       ���E	,L�%[��A��	*

A2S/average_reward_1��8C�� &,       ���E	�,�%[��AȞ	*

A2S/average_reward_1�^B��5�,       ���E	�Gg&[��A��	*

A2S/average_reward_1Xs�C��+,       ���E	�r&[��A��	*

A2S/average_reward_1�x�At8�2,       ���E	��3'[��A��	*

A2S/average_reward_1��C_�\,       ���E	n([��A��	*

A2S/average_reward_1�F�CX#�,       ���E	�'([��A��	*

A2S/average_reward_1V��A��^S,       ���E	�'([��A֪	*

A2S/average_reward_1e��A� ]R,       ���E	�q)[��A��	*

A2S/average_reward_1�f"DF���,       ���E	�*[��A�	*

A2S/average_reward_1.Z*D��14,       ���E	�*�+[��A��	*

A2S/average_reward_1�'�C�s��,       ���E	���+[��A��	*

A2S/average_reward_1m��B�
�3,       ���E	�w?-[��A��	*

A2S/average_reward_1,DDe�$,       ���E	���-[��A��	*

A2S/average_reward_1�ԘC�� >,       ���E	�$;.[��A��	*

A2S/average_reward_1�FC�k�,       ���E	b�D.[��A��	*

A2S/average_reward_1���A.t',       ���E	�,H/[��A��	*

A2S/average_reward_17�CIBD�,       ���E	
�h/[��A��	*

A2S/average_reward_1�[PB���,       ���E	��z/[��A��	*

A2S/average_reward_1~I�A-#��,       ���E	�f�/[��A��	*

A2S/average_reward_1�B�.��,       ���E	�h0[��A��	*

A2S/average_reward_1�Q�C!C�^,       ���E	�v0[��A��	*

A2S/average_reward_1�[�A\.W,       ���E	��0[��A��	*

A2S/average_reward_1�YB �{�,       ���E	�Ҙ0[��A��	*

A2S/average_reward_1<�aB��ؑ,       ���E	ۣ0[��A��	*

A2S/average_reward_1�W�AQ'�,       ���E	1�0[��A��	*

A2S/average_reward_1
��@��,       ���E	,��0[��A��	*

A2S/average_reward_1Y��A�m��x       ��!�	z��9[��A��	*i

A2S/kl-lL:

A2S/policy_network_losst�

A2S/value_network_loss@PWC

A2S/q_network_lossZ�VC��SN,       ���E	5U&;[��A��	*

A2S/average_reward_1MD�w��,       ���E	:R0;[��A��	*

A2S/average_reward_1(b�A�>4�,       ���E	!�7;[��A��	*

A2S/average_reward_1��A$�V,       ���E	���<[��A��	*

A2S/average_reward_1�L2D/_Z,       ���E	ϧ<[��A��	*

A2S/average_reward_1s�	B��T,       ���E	��N=[��A��	*

A2S/average_reward_1�ܠC3'Lz,       ���E	Y��=[��A��	*

A2S/average_reward_18
�B�:��,       ���E	1��=[��A��	*

A2S/average_reward_1>%zA��C�,       ���E	�s�=[��A��	*

A2S/average_reward_1�B���	,       ���E	q{�=[��A��	*

A2S/average_reward_1��B̆��,       ���E	��=[��A��	*

A2S/average_reward_1��B19,       ���E	D�=[��A��	*

A2S/average_reward_1g��A���,       ���E	E� >[��A��	*

A2S/average_reward_10��BR��,       ���E	ԍ�>[��A��	*

A2S/average_reward_16GmC���,       ���E	x��?[��A��	*

A2S/average_reward_1%�Dg��Q,       ���E	�7GA[��A��
*

A2S/average_reward_1ֹ/D
@,       ���E	��B[��A�
*

A2S/average_reward_1�4D~�,       ���E	}}�B[��A��
*

A2S/average_reward_1%o�A�,       ���E	��C[��A�
*

A2S/average_reward_1M�0D��l,       ���E	��D[��A��
*

A2S/average_reward_1Y��B<r@�x       ��!�	�gM[��A��
*i

A2S/kl2�:

A2S/policy_network_loss4_Ŀ

A2S/value_network_lossO�C

A2S/q_network_loss�N�C�\c4,       ���E	\s�M[��A��
*

A2S/average_reward_1�V�B����,       ���E	ٔ�M[��A��
*

A2S/average_reward_1�LBJ#j{,       ���E	9�M[��Aܖ
*

A2S/average_reward_1�� BA�o?,       ���E	 5�M[��A��
*

A2S/average_reward_1�yA����,       ���E	΄�M[��Aؗ
*

A2S/average_reward_1+&Bz�,       ���E	@6O[��A��
*

A2S/average_reward_1^�1D�I=C,       ���E	um�O[��A��
*

A2S/average_reward_1Z�RC��|,       ���E	OSHP[��A�
*

A2S/average_reward_1��CY@�,       ���E	}`P[��A��
*

A2S/average_reward_1N B!�,       ���E	�qP[��Aɦ
*

A2S/average_reward_1��e@��D,       ���E	і�P[��A��
*

A2S/average_reward_1�iQB�S�Z,       ���E	
+�P[��A��
*

A2S/average_reward_1��,@�՞2,       ���E	��P[��A٧
*

A2S/average_reward_1|A'�Y,       ���E	���P[��Aè
*

A2S/average_reward_1G{+B�f�_,       ���E	(�P[��A��
*

A2S/average_reward_1��>B��:�,       ���E	ڡ�Q[��A��
*

A2S/average_reward_1���C���,       ���E	#�Q[��A�
*

A2S/average_reward_1i�B�0�,       ���E	�:�Q[��A��
*

A2S/average_reward_1��JC���,       ���E	��R[��A�
*

A2S/average_reward_1��Bur��,       ���E	8R[��A��
*

A2S/average_reward_1��d@�-",       ���E	��)R[��A��
*

A2S/average_reward_1N��Ao=V,       ���E	�RxR[��A��
*

A2S/average_reward_1��B��,       ���E	7�R[��A��
*

A2S/average_reward_1&-B��>,       ���E	��R[��A��
*

A2S/average_reward_1��2C�a��,       ���E	�/S[��AƵ
*

A2S/average_reward_1�P�BI۫,       ���E	(�:S[��A�
*

A2S/average_reward_1\m�A��H,       ���E	�\fS[��AӶ
*

A2S/average_reward_13j�B�'Ph,       ���E	ǷHT[��A��
*

A2S/average_reward_1Z��C�u�,       ���E	Df\T[��Aػ
*

A2S/average_reward_1��4B�}�),       ���E	�<�U[��A��
*

A2S/average_reward_1�*Df0�,       ���E	8ܤV[��A��
*

A2S/average_reward_1V,�C��~�,       ���E	0$�V[��A��
*

A2S/average_reward_1;�FC9�F�,       ���E	~�W[��A��
*

A2S/average_reward_10��C�q$�,       ���E	jJ�W[��A��
*

A2S/average_reward_1�-�A�",       ���E	f3�X[��A��
*

A2S/average_reward_1��*D0h,       ���E	�l;Z[��A��
*

A2S/average_reward_1��*D,[�,       ���E	���Z[��A��
*

A2S/average_reward_1x�C7��1,       ���E	NZ�Z[��A��
*

A2S/average_reward_1���A�z&�,       ���E	�l[[��A��
*

A2S/average_reward_1��|Ca�,       ���E	pH�[[��A��
*

A2S/average_reward_1���B�<%,       ���E	m�l\[��A��
*

A2S/average_reward_1@c�C�2~x       ��!�	���e[��A��
*i

A2S/kl�!:

A2S/policy_network_loss�b��

A2S/value_network_loss��`C

A2S/q_network_loss��]CQ��a,       ���E	��[g[��A��
*

A2S/average_reward_1�~3D5��,       ���E	6�vg[��A��
*

A2S/average_reward_19��AS���,       ���E	9χg[��A��
*

A2S/average_reward_1�ZAW]�,       ���E	L��h[��A��
*

A2S/average_reward_1W/De�i,       ���E	��h[��A��
*

A2S/average_reward_1_�"B{+�w,       ���E	�L�h[��A��
*

A2S/average_reward_1%pkB6��,       ���E	{��h[��A��
*

A2S/average_reward_1e)�A�|5�,       ���E	W,�h[��A��
*

A2S/average_reward_1��dB*q~�,       ���E	��i[��A��*

A2S/average_reward_1�[�C���,       ���E	�w�i[��A��*

A2S/average_reward_1c�=B���y,       ���E	���i[��A߄*

A2S/average_reward_1�R@�[�%,       ���E		itj[��A��*

A2S/average_reward_1Y
�C=�,       ���E	*~j[��A��*

A2S/average_reward_1n%�Aho��,       ���E	�şj[��A��*

A2S/average_reward_1{.B3�*#,       ���E	qĳj[��A��*

A2S/average_reward_1��EB���,       ���E	$��j[��A�*

A2S/average_reward_1���B2	��,       ���E	|
k[��A��*

A2S/average_reward_1%!?B��k�,       ���E	��*k[��A��*

A2S/average_reward_1
<�B2D
,       ���E	��;k[��A�*

A2S/average_reward_1��A�� ,       ���E	CHk[��A��*

A2S/average_reward_1��A���R,       ���E	�@ck[��A�*

A2S/average_reward_1�ITB�b��x       ��!�	*o"t[��A�*i

A2S/kl7�R;

A2S/policy_network_lossw���

A2S/value_network_loss!�C

A2S/q_network_loss�;�C[XS,       ���E	�pu[��A˕*

A2S/average_reward_1�7D��&:,       ���E	&�v[��A��*

A2S/average_reward_1	:Dɮ܋,       ���E	5�*x[��A��*

A2S/average_reward_1=G:D
��,       ���E	(y�x[��A��*

A2S/average_reward_1���CL-��,       ���E	��x[��A¨*

A2S/average_reward_1�I?A�Y,       ���E	N��x[��A��*

A2S/average_reward_1k*�A�Ͻ9,       ���E	�z[��A�*

A2S/average_reward_1m�:D�5�
,       ���E	'z[��A��*

A2S/average_reward_1�*B܍B$,       ���E	�<z[��AԱ*

A2S/average_reward_1���Ab:ϵ,       ���E	��{[��A��*

A2S/average_reward_1��;D����,       ���E	�ݙ{[��A��*

A2S/average_reward_1��DAB[��,       ���E	�\�{[��Aѻ*

A2S/average_reward_1�*C�@,       ���E	���{[��A��*

A2S/average_reward_1���A[	�^,       ���E	��{[��A��*

A2S/average_reward_1-%B4���,       ���E	XE|[��A�*

A2S/average_reward_1���A×{b,       ���E	n�}[��A��*

A2S/average_reward_1�;Dp7�,       ���E	��}[��A��*

A2S/average_reward_1B�(B_��t,       ���E	��[��A��*

A2S/average_reward_1�c6Dkg�),       ���E	��|[��A��*

A2S/average_reward_1��jCpN�,       ���E	�S�[��A��*

A2S/average_reward_1.�SC!ħ�,       ���E	��[��A��*

A2S/average_reward_1��BG�OK,       ���E	Y[��A��*

A2S/average_reward_1�]�C֧^�,       ���E	����[��A��*

A2S/average_reward_1<eB��},       ���E	�憁[��A��*

A2S/average_reward_1P�C�~�,       ���E	�걂[��A��*

A2S/average_reward_1�-D�_Yj,       ���E	ؿ��[��A��*

A2S/average_reward_1NBԨzS,       ���E	:ǂ[��A��*

A2S/average_reward_1L��A��,       ���E	�I�[��A��*

A2S/average_reward_1���CO؍,       ���E	�1o�[��A��*

A2S/average_reward_1��B#��K,       ���E	�0W�[��A��*

A2S/average_reward_1��C���,       ���E	s~�[��A��*

A2S/average_reward_1s�B\+��,       ���E	�6��[��A��*

A2S/average_reward_1H*�C��,       ���E	�#i�[��A��*

A2S/average_reward_1�]C{ӎ ,       ���E		ـ�[��A��*

A2S/average_reward_17XB���,       ���E	Qw�[��A��*

A2S/average_reward_1��C�|>\,       ���E	}�&�[��A��*

A2S/average_reward_1֭SBw�y;,       ���E	6j�[��A��*

A2S/average_reward_1R�B`�{,       ���E	{��[��A��*

A2S/average_reward_1)�=B+��,       ���E	�٣�[��A��*

A2S/average_reward_1��AM�S,       ���E	�ͻ�[��A��*

A2S/average_reward_1Sw�Am��K,       ���E	?��[��A��*

A2S/average_reward_1�4C���,       ���E	M��[��A��*

A2S/average_reward_1�*B���,       ���E	Ă
�[��A�*

A2S/average_reward_1!�C���"x       ��!�	;��[��A�*i

A2S/kl���:

A2S/policy_network_lossG��

A2S/value_network_loss�aC

A2S/q_network_loss|�dC?���,       ���E	[�v�[��Aۈ*

A2S/average_reward_1��4DJ�M�,       ���E	�'��[��A��*

A2S/average_reward_1�A��",       ���E	M]��[��A��*

A2S/average_reward_1���A�@8+,       ���E	��^�[��A��*

A2S/average_reward_1���C#���,       ���E	�,��[��A��*

A2S/average_reward_1c[3DL��",       ���E	���[��A��*

A2S/average_reward_1��C���,       ���E	|�[��A��*

A2S/average_reward_1�8DȈ°,       ���E	K	�[��Aܣ*

A2S/average_reward_1~�B10�,       ���E	��f�[��Aī*

A2S/average_reward_1�I6D���,       ���E	�
r�[��A�*

A2S/average_reward_1Ha�A�?�?,       ���E	�A�[��A�*

A2S/average_reward_1�d�C]�"�,       ���E	ρL�[��A��*

A2S/average_reward_1�=�AU���,       ���E	�8��[��A�*

A2S/average_reward_1�h6D�O,       ���E	H|��[��Aټ*

A2S/average_reward_1��C.�Q,       ���E	���[��A��*

A2S/average_reward_1 ��AoY�,       ���E		���[��A��*

A2S/average_reward_1\�4D���{,       ���E	����[��A��*

A2S/average_reward_1j 9D}�,�,       ���E	cȡ�[��A��*

A2S/average_reward_1���B��^�,       ���E	w+�[��A��*

A2S/average_reward_1�{C�]7D,       ���E	1���[��A��*

A2S/average_reward_1��C-��,       ���E	<�[��A��*

A2S/average_reward_1��4D7��,       ���E	�C��[��A��*

A2S/average_reward_11�fC��x       ��!�	c#��[��A��*i

A2S/kl�oM;

A2S/policy_network_lossh�n�

A2S/value_network_loss��rC

A2S/q_network_loss,HpC1F=,       ���E	��ӫ[��A��*

A2S/average_reward_1�{^BT���,       ���E	kx>�[��A��*

A2S/average_reward_1UC0+3�,       ���E	jyJ�[��A��*

A2S/average_reward_1�<B%�:;,       ���E	�_�[��A��*

A2S/average_reward_1�e#B�V,       ���E	Ҙl�[��A��*

A2S/average_reward_1��B�ر,       ���E	�({�[��A��*

A2S/average_reward_1�B��ֲ,       ���E	0���[��A��*

A2S/average_reward_1�A�[ib,       ���E	��¬[��A��*

A2S/average_reward_1�=C���,       ���E	n;Ϭ[��A��*

A2S/average_reward_1}-�Ax��,       ���E	�x�[��A��*

A2S/average_reward_1?�C#��,       ���E	�|�[��A��*

A2S/average_reward_1%P�A�/?Y,       ���E	*+��[��A��*

A2S/average_reward_1�D�AD9��,       ���E	�ɮ[��A��*

A2S/average_reward_1տ6Di�P,       ���E	�2Ԯ[��A��*

A2S/average_reward_1!z�A Ɖ,       ���E	���[��A��*

A2S/average_reward_1�}�A�B,       ���E	���[��A��*

A2S/average_reward_1\�A�AU�,       ���E	�h��[��A��*

A2S/average_reward_1*m D����,       ���E	eh�[��A��*

A2S/average_reward_1g�&B�ԭ,       ���E	+�<�[��A��*

A2S/average_reward_1��/D��,,       ���E	�/H�[��A��*

A2S/average_reward_1d��AY��o,       ���E	`8[�[��A��*

A2S/average_reward_1��B9��C,       ���E	b�c�[��A��*

A2S/average_reward_1��A�ex       ��!�	W09�[��A��*i

A2S/kl�R{;

A2S/policy_network_loss��V�

A2S/value_network_loss�ghC

A2S/q_network_lossӏjC����,       ���E	 �H�[��A��*

A2S/average_reward_1;33B �<�,       ���E	�P�[��A��*

A2S/average_reward_1�j�A���9,       ���E	��\�[��A��*

A2S/average_reward_1$B�'�,       ���E	��c�[��A��*

A2S/average_reward_1���A����,       ���E	͂k�[��A��*

A2S/average_reward_1m��AL���,       ���E	syv�[��A��*

A2S/average_reward_1B�A�,       ���E	��{�[��A��*

A2S/average_reward_1�C�A�#�,       ���E	C���[��A��*

A2S/average_reward_1�_�AI�А,       ���E	D\��[��Aƀ*

A2S/average_reward_1SB"9�,       ���E	�X��[��A�*

A2S/average_reward_1�BS���,       ���E	���[��AԈ*

A2S/average_reward_12�<D���c,       ���E	����[��A�*

A2S/average_reward_1���A��2D,       ���E	�Xk�[��AӐ*

A2S/average_reward_1)�9DM���,       ���E	��s�[��A�*

A2S/average_reward_1�K�Aq�,       ���E	v�}�[��A��*

A2S/average_reward_1U}�A�5�z,       ���E	�Ɗ�[��A��*

A2S/average_reward_1�ZB���,       ���E	X�[��A��*

A2S/average_reward_1�";D���E,       ���E	lW�[��AǙ*

A2S/average_reward_1U'�A����,       ���E	E�G�[��A*

A2S/average_reward_1K7CC�`1,       ���E	I�O�[��Aܛ*

A2S/average_reward_1�v�A.�,       ���E	v�\�[��A��*

A2S/average_reward_17<�A�%�,       ���E	��a�[��A��*

A2S/average_reward_1;q�AЗ�x       ��!�	p�[��A��*i

A2S/kl�,�;

A2S/policy_network_loss�mA�

A2S/value_network_loss��C

A2S/q_network_loss�Q�C�]�,       ���E	�}D�[��A��*

A2S/average_reward_1P�C2���,       ���E	@UU�[��A��*

A2S/average_reward_1ٵ�A
�,       ���E	1�]�[��A��*

A2S/average_reward_1���A�T*e,       ���E	B���[��A��*

A2S/average_reward_11��B��l�,       ���E	TN��[��Aɟ*

A2S/average_reward_1�!BU6,       ���E	���[��A�*

A2S/average_reward_1!S�A�u��,       ���E	^���[��A��*

A2S/average_reward_16kB��,       ���E	�o��[��A��*

A2S/average_reward_1�f�A�,�,       ���E	���[��A��*

A2S/average_reward_1�8�A	BQ�,       ���E	���[��Aՠ*

A2S/average_reward_1l��Ae���,       ���E	�K��[��A��*

A2S/average_reward_1�_�B>V�,       ���E	����[��A̡*

A2S/average_reward_1���A�-�,       ���E	\6��[��A��*

A2S/average_reward_1s�B���,       ���E	[��[��A��*

A2S/average_reward_1���A�O%n,       ���E	��	�[��A��*

A2S/average_reward_1���A6��,       ���E	��[��A�*

A2S/average_reward_1̞xB.>,       ���E	L)*�[��A��*

A2S/average_reward_1�
B�:K�,       ���E	��I�[��A�*

A2S/average_reward_1��B���;,       ���E	�`�[��A��*

A2S/average_reward_1��PB�K�,       ���E	�,�[��A��*

A2S/average_reward_1!z�C
��,       ���E	'k5�[��A��*

A2S/average_reward_1AT�A!�,       ���E	��:�[��A��*

A2S/average_reward_1)�Av�,       ���E	�OL�[��A��*

A2S/average_reward_1ɧB��zz,       ���E	%BV�[��A��*

A2S/average_reward_1=��C�Mv�,       ���E	TF	�[��A��*

A2S/average_reward_1��C	'�k,       ���E	�۬�[��A׶*

A2S/average_reward_1�6�C?0,       ���E	�*��[��A��*

A2S/average_reward_1�0D���,       ���E	�v��[��A�*

A2S/average_reward_1�׽A��QX,       ���E	EM�[��A��*

A2S/average_reward_1�BAmI$,       ���E	N~[�[��A��*

A2S/average_reward_1ګD�� ,       ���E	6���[��A��*

A2S/average_reward_1�C!�{,       ���E	���[��A��*

A2S/average_reward_1%�A_�
,       ���E	/@�[��A��*

A2S/average_reward_1�*%By�-,       ���E	�w#�[��A��*

A2S/average_reward_1C��A[�h,       ���E	�r{�[��A��*

A2S/average_reward_1�8C���,       ���E	�&��[��A��*

A2S/average_reward_12�+D�
F,       ���E	��[��A��*

A2S/average_reward_1S�:C�J}�,       ���E	o�&�[��A��*

A2S/average_reward_12BE�Z�,       ���E	�t3�[��A��*

A2S/average_reward_1�1HAѲ_�,       ���E	6
��[��A��*

A2S/average_reward_1�+D%l��,       ���E	�|�[��A��*

A2S/average_reward_1�1D���_,       ���E	�6�[��A��*

A2S/average_reward_1��BrJ�k,       ���E	z��[��A��*

A2S/average_reward_1�]CF��2,       ���E	�4�[��A��*

A2S/average_reward_1�..D �,       ���E		UC�[��A��*

A2S/average_reward_1칔AA���,       ���E	�|b�[��A��*

A2S/average_reward_1m	D�6��,       ���E	����[��A��*

A2S/average_reward_1g�CL�]�,       ���E	�/�[��A��*

A2S/average_reward_1 �HB� �,       ���E	���[��A��*

A2S/average_reward_1(Z|C�{�,       ���E	$���[��A��*

A2S/average_reward_1��Aþi�,       ���E	^��[��A��*

A2S/average_reward_1�+�@S�@,       ���E	�t��[��A߇*

A2S/average_reward_1h�)D<F=,       ���E	��
�[��A��*

A2S/average_reward_1�4�A	��{,       ���E	·�[��A��*

A2S/average_reward_1nf�A@�!,       ���E	!6*�[��Aƈ*

A2S/average_reward_1�Bx"�,       ���E	��4�[��A�*

A2S/average_reward_1b�AI(��,       ���E	�k<�[��A��*

A2S/average_reward_1a7�Ai�',       ���E	LQ��[��A�*

A2S/average_reward_1[�-DaER,       ���E	���[��AИ*

A2S/average_reward_1�/DXQR�,       ���E	�v.�[��Aϙ*

A2S/average_reward_1;��B3cm,       ���E	{B�[��A��*

A2S/average_reward_1/�	B1�R,       ���E	�+��[��A��*

A2S/average_reward_1�:�CI��&,       ���E	r~R�[��A��*

A2S/average_reward_12G+DF7&,       ���E	��i�[��Aܥ*

A2S/average_reward_1v&B|�&�,       ���E	����[��Aĭ*

A2S/average_reward_1X=)Ds���,       ���E	����[��A��*

A2S/average_reward_1N�iB���,       ���E	j���[��A��*

A2S/average_reward_19�B�k,       ���E	E���[��A��*

A2S/average_reward_1IiAY��~,       ���E	tH��[��Aǳ*

A2S/average_reward_1�k�C$t�,       ���E	#)�[��A��*

A2S/average_reward_1HL-D|�m�,       ���E	�>�[��A�*

A2S/average_reward_1�" C�,p,       ���E	�H��[��A��*

A2S/average_reward_1�-D�#��,       ���E	#s��[��A��*

A2S/average_reward_1���C$
(�,       ���E	���[��A��*

A2S/average_reward_1��BV��,       ���E	n)�[��A��*

A2S/average_reward_1���A���,       ���E	�25�[��A��*

A2S/average_reward_1#%�A�g�p,       ���E	��[��A��*

A2S/average_reward_1:�iC����,       ���E	sӪ�[��A��*

A2S/average_reward_1��A^њ.,       ���E	�%C�[��A��*

A2S/average_reward_1k]�C�+�,       ���E	"�F�[��A��*

A2S/average_reward_1�
�A�܃@,       ���E	V=y�[��A��*

A2S/average_reward_1��D�D��,       ���E	cŘ�[��A��*

A2S/average_reward_1�O�A�,,       ���E	�x��[��A��*

A2S/average_reward_1�=An���,       ���E	ƪ��[��A��*

A2S/average_reward_1�&�Aaf,       ���E	�b�[��A��*

A2S/average_reward_1�Cw�:',       ���E	�g�[��A��*

A2S/average_reward_1���AW"�F,       ���E	�al�[��A��*

A2S/average_reward_1�8�A@�� ,       ���E	R�/�[��A��*

A2S/average_reward_1�q�C���,       ���E	�<�[��A��*

A2S/average_reward_1��B��:,       ���E	���[��A��*

A2S/average_reward_1p �C�1�,       ���E	%�e�[��A��*

A2S/average_reward_1��Ctj�,       ���E	��m�[��A��*

A2S/average_reward_1�B7ǭ�,       ���E	�Iu�[��A��*

A2S/average_reward_1�E�A��ֳ,       ���E	2���[��A��*

A2S/average_reward_1���Ao8�Ux       ��!�	H���[��A��*i

A2S/kl��n=

A2S/policy_network_loss{�

A2S/value_network_loss�%fC

A2S/q_network_loss��hC��$�,       ���E	����[��A��*

A2S/average_reward_1w��Cx%<�,       ���E	J#�[��A��*

A2S/average_reward_1@�BG�E,       ���E	��(�[��A��*

A2S/average_reward_1�)�Ab��,       ���E	%yt�[��A��*

A2S/average_reward_1Y2D<{4,       ���E	 0��[��A��*

A2S/average_reward_1e#D�6eE,       ���E	����[��A��*

A2S/average_reward_1�g#Bc8$,       ���E	���[��A��*

A2S/average_reward_1�/D)A r,       ���E	�5�[��A��*

A2S/average_reward_1B��B�~̉,       ���E	�bC�[��A��*

A2S/average_reward_1��A̤��,       ���E	/f��[��A��*

A2S/average_reward_1��2D>)�,       ���E	�$��[��A��*

A2S/average_reward_10�A��%,       ���E	rx��[��AҐ*

A2S/average_reward_1kTBv�A�,       ���E	���[��A��*

A2S/average_reward_1���B���g,       ���E	���[��A��*

A2S/average_reward_1{��C���,       ���E	����[��A��*

A2S/average_reward_1��A\��,       ���E	o���[��A�*

A2S/average_reward_1��xB}��
,       ���E	�;��[��A��*

A2S/average_reward_1�_!A��u�,       ���E	ީ��[��A��*

A2S/average_reward_1y��C)f�\,       ���E	�,��[��Aǜ*

A2S/average_reward_1�h�A�k��,       ���E	Ps��[��A��*

A2S/average_reward_1�4DCve�,       ���E	����[��Aڤ*

A2S/average_reward_1j'B��tG,       ���E	0���[��A��*

A2S/average_reward_1a��Cy�q�,       ���E	�\��A��*

A2S/average_reward_1�1D8� �,       ���E	8/\��A��*

A2S/average_reward_1�f�A|��,       ���E	{�k\��A��*

A2S/average_reward_1Cwc�x       ��!�	��6\��A��*i

A2S/kllC�8

A2S/policy_network_lossd���

A2S/value_network_lossY=�C

A2S/q_network_loss��}C��?,       ���E	��R\��A�*

A2S/average_reward_1��B����,       ���E	u:�\��A��*

A2S/average_reward_1�Ds�G,       ���E	��\��AϹ*

A2S/average_reward_1�� Bm��,       ���E	�E4\��A��*

A2S/average_reward_1���C��4',       ���E	�+�\��A��*

A2S/average_reward_1U0D�8 �,       ���E	��\��A��*

A2S/average_reward_1q�4DY:��,       ���E	�\\��A��*

A2S/average_reward_1���A���,       ���E	��\��A��*

A2S/average_reward_1�4�A���,       ���E	�k\��A��*

A2S/average_reward_1��UCSDJ�,       ���E	�	�\��A��*

A2S/average_reward_1�p5D��	�,       ���E	N��\��A��*

A2S/average_reward_1�G,B*�@Y,       ���E	<��\��A��*

A2S/average_reward_1�'A�~3�,       ���E	���\��A��*

A2S/average_reward_1��UA���,       ���E	���\��A��*

A2S/average_reward_1��xA�H�,       ���E	��\��A��*

A2S/average_reward_1[�Bd��,       ���E	��\��A��*

A2S/average_reward_1���A�S�,       ���E	��e\��A��*

A2S/average_reward_1S5Dk�%,       ���E	]�\��A��*

A2S/average_reward_1�0DSj�J,       ���E	��R\��A��*

A2S/average_reward_1'�7D{�C�,       ���E	0�\��A��*

A2S/average_reward_1���Ca[��,       ���E	v -\��A��*

A2S/average_reward_1�_�Ap3,       ���E	h7\��A��*

A2S/average_reward_1���AP�~�,       ���E	'�C\��A��*

A2S/average_reward_1�3$B���,       ���E	+�\��A��*

A2S/average_reward_1��2Dt/{�,       ���E	 P
\��A�*

A2S/average_reward_1O`)D��R,       ���E	O21\��A�*

A2S/average_reward_1��EBT�,       ���E	nMC\��A��*

A2S/average_reward_1�SB�Q��,       ���E	���\��A��*

A2S/average_reward_1K�)D�LA,       ���E	%�\��Aȏ*

A2S/average_reward_1�i%BW�h�,       ���E	�B\��A��*

A2S/average_reward_1@}*DŜ��,       ���E	r�1\��A�*

A2S/average_reward_1'�2B!�a,       ���E	�@\��A��*

A2S/average_reward_1D!�A�~#�,       ���E	��F\��A��*

A2S/average_reward_1�ԬA�8,       ���E	�jP\��Aߘ*

A2S/average_reward_1��gAA��,       ���E	�g�\��Aޚ*

A2S/average_reward_10ACsJ�3,       ���E	 X�\��A��*

A2S/average_reward_1��DBa>,       ���E	� \��A��*

A2S/average_reward_1�f*D*�NY,       ���E	��# \��Aթ*

A2S/average_reward_1o�B��.,       ���E	�p� \��A̫*

A2S/average_reward_1�-C��V�,       ���E	��!\��A��*

A2S/average_reward_1D�D��1�,       ���E	kB#\��A��*

A2S/average_reward_1d�D��O�,       ���E	~�#\��A��*

A2S/average_reward_1��A��,       ���E	�r)#\��A��*

A2S/average_reward_1�aTA���C,       ���E	�2#\��A��*

A2S/average_reward_1�!�AC���,       ���E	al#\��A��*

A2S/average_reward_1��C�3�,       ���E	}X�$\��A��*

A2S/average_reward_1�L+DjFD^,       ���E	��$\��A��*

A2S/average_reward_1�b:BK�]�,       ���E	}΍%\��A��*

A2S/average_reward_15D$֮�,       ���E	U��%\��A��*

A2S/average_reward_1s�A�X(�,       ���E	�%�&\��A��*

A2S/average_reward_1�+D���,       ���E	��(\��A��*

A2S/average_reward_1J{*D��Ex       ��!�	
�Z2\��A��*i

A2S/kl�1�6

A2S/policy_network_lossd�ȿ

A2S/value_network_loss$�[C

A2S/q_network_loss�^C�;�w,       ���E	�ż2\��A��*

A2S/average_reward_1�8HC���,       ���E	���3\��A��*

A2S/average_reward_1��C��f�,       ���E	b��3\��A��*

A2S/average_reward_1i�B�NF#,       ���E	�
�3\��A��*

A2S/average_reward_1n��A,N�,,       ���E	�5\��A��*

A2S/average_reward_1��4D�A��,       ���E	$�6\��A��*

A2S/average_reward_1��3D�ػ:,       ���E	L0�6\��A��*

A2S/average_reward_1pBE���,       ���E	!��6\��A��*

A2S/average_reward_1���A�	��,       ���E	%��7\��A��*

A2S/average_reward_1�8D���,       ���E	���7\��A��*

A2S/average_reward_1NZ�AH�S~,       ���E	g�8\��A��*

A2S/average_reward_1�Z�Cۆ�Q,       ���E	���8\��A��*

A2S/average_reward_1�%^Bh�o�,       ���E	� �9\��A��*

A2S/average_reward_1JS6D$I��,       ���E	�8d:\��AǊ*

A2S/average_reward_1�A�C��/,       ���E	,+l:\��A܊*

A2S/average_reward_1t"�A~��,       ���E	oض:\��A��*

A2S/average_reward_1�[Cd �,       ���E	A��:\��Aʌ*

A2S/average_reward_1Jc�A��Q�,       ���E	�0<\��A��*

A2S/average_reward_17K7D7mk�,       ���E	�0<\��A�*

A2S/average_reward_1j�B�"9K,       ���E	�[�<\��A��*

A2S/average_reward_1��C��y,       ���E	3��<\��A��*

A2S/average_reward_1-�	@���],       ���E	�!�<\��Aј*

A2S/average_reward_1w��A�U�,       ���E	{�>\��A��*

A2S/average_reward_1\"3DU$P�,       ���E	�*
?\��A��*

A2S/average_reward_1�C��,       ���E	,6?\��A��*

A2S/average_reward_1�[�A�+,       ���E	!(/?\��A�*

A2S/average_reward_10[�B���x       ��!�	Q��H\��A�*i

A2S/kl~2:

A2S/policy_network_loss���

A2S/value_network_loss@ӅC

A2S/q_network_loss_�C@��&,       ���E	�%TJ\��Aٮ*

A2S/average_reward_1��6D�U�,       ���E	��fJ\��A��*

A2S/average_reward_1�
?B<�k,       ���E	)�zJ\��Aۯ*

A2S/average_reward_1$�#B���N,       ���E	4�J\��A��*

A2S/average_reward_1�*A,�",       ���E	��J\��A��*

A2S/average_reward_1� �A�*,       ���E	�y�J\��Aְ*

A2S/average_reward_1z7Bs܍�,       ���E	���J\��A��*

A2S/average_reward_1W��A� ,       ���E	��K\��A��*

A2S/average_reward_1&�DY3�O,       ���E	���K\��A��*

A2S/average_reward_1��A�0z=,       ���E	��L\��A��*

A2S/average_reward_1OCi��,       ���E	}hL\��A��*

A2S/average_reward_1K��AB�,       ���E	�heM\��A��*

A2S/average_reward_1n�6D]7X,       ���E	��M\��A��*

A2S/average_reward_1���B�sz�,       ���E	1�M\��A��*

A2S/average_reward_1���AtS,       ���E	6�M\��A��*

A2S/average_reward_1'��A�],       ���E	<a�M\��A��*

A2S/average_reward_1?BԘ��,       ���E	=0�N\��A��*

A2S/average_reward_1P�CEi`,       ���E	b��N\��A��*

A2S/average_reward_1�L�B�5*,       ���E	W#�N\��A��*

A2S/average_reward_1��B��(�,       ���E	-GP\��A��*

A2S/average_reward_17�9Dj��L,       ���E	5��Q\��A��*

A2S/average_reward_1�:DQy,       ���E	�&zR\��A��*

A2S/average_reward_16V�C��X�,       ���E	w�<S\��A��*

A2S/average_reward_1ba�CV=��,       ���E	lIS\��A��*

A2S/average_reward_166�AdLĖ,       ���E	���S\��A��*

A2S/average_reward_1��	C�^�,       ���E	�N�S\��A��*

A2S/average_reward_1�B���"x       ��!�	�\\��A��*i

A2S/kl��:

A2S/policy_network_lossR��

A2S/value_network_loss6*�C

A2S/q_network_loss���C��Q�,       ���E	�-�]\��A��*

A2S/average_reward_1�f�Clf�k,       ���E	�,�]\��A��*

A2S/average_reward_1\�B�U��,       ���E	���]\��A��*

A2S/average_reward_1�A'�',       ���E	s��]\��A��*

A2S/average_reward_1�C����,       ���E	�^\��A��*

A2S/average_reward_1C�B���,       ���E	��-_\��A��*

A2S/average_reward_1��DN�3N,       ���E	v�F_\��A��*

A2S/average_reward_1���AH�2y,       ���E	[TR_\��A��*

A2S/average_reward_1/��A_4��,       ���E	1�Y_\��A��*

A2S/average_reward_1���A�ķ�,       ���E	vf_\��A��*

A2S/average_reward_16�A}�,       ���E	^��`\��A��*

A2S/average_reward_1��9D$���,       ���E	&�?a\��A��*

A2S/average_reward_1W�oC�1�,       ���E	���a\��A��*

A2S/average_reward_1Hs�Cgi,       ���E	�o b\��A��*

A2S/average_reward_1+#+BEū�,       ���E	UwSc\��A��*

A2S/average_reward_1�:D�_ET,       ���E	h�ic\��A��*

A2S/average_reward_1���Bf{�,       ���E	#�d\��A��*

A2S/average_reward_1�/D���W,       ���E	u3�e\��A��*

A2S/average_reward_1�L!DX;C,       ���E	*_�e\��A��*

A2S/average_reward_1~�EB-5�,       ���E	��e\��A�*

A2S/average_reward_1:	C�9:,       ���E	���e\��A��*

A2S/average_reward_1:�B��g,       ���E	K�f\��AƖ*

A2S/average_reward_1�A�c�,       ���E	BPf\��A��*

A2S/average_reward_1Z�Be�Q,       ���E	��Ug\��Aݞ*

A2S/average_reward_1��9DF65�,       ���E	���g\��A��*

A2S/average_reward_1�ڜC}CY	,       ���E	� h\��A��*

A2S/average_reward_1��B=�,       ���E	��_i\��A��*

A2S/average_reward_1M*+Dgm��,       ���E	�z�j\��A��*

A2S/average_reward_1�*D���,       ���E	��j\��A��*

A2S/average_reward_1vo�A@�,       ���E	v�l\��A��*

A2S/average_reward_1��,DGr�{,       ���E	�fXl\��A��*

A2S/average_reward_1�f.C�n�,       ���E	��l\��A��*

A2S/average_reward_1i��C�}v4,       ���E	�AMn\��A��*

A2S/average_reward_1��)D�7q�,       ���E	gu�n\��A��*

A2S/average_reward_1��gC ,?,       ���E	8��n\��A��*

A2S/average_reward_1���B�s],       ���E	gC-p\��A��*

A2S/average_reward_1��DG�d,       ���E	�/�p\��A��*

A2S/average_reward_16��C_kn,       ���E	��
r\��A��*

A2S/average_reward_1E�Dym�,       ���E	��r\��A��*

A2S/average_reward_1���@E@0,       ���E	I�ts\��A��*

A2S/average_reward_1E'D��,       ���E	�ߒs\��A��*

A2S/average_reward_19�B4En,       ���E	�m�s\��A��*

A2S/average_reward_1�O0C�Qb,       ���E	�+=u\��A��*

A2S/average_reward_1q.D9�,       ���E	�̵u\��A��*

A2S/average_reward_1�ZFC-+��,       ���E	��v\��A��*

A2S/average_reward_1ɱ�Cd��&,       ���E	u�w\��A��*

A2S/average_reward_17�(D���,       ���E	��[y\��A��*

A2S/average_reward_1.D��9,       ���E	���z\��A��*

A2S/average_reward_1�*D$�{',       ���E	��z\��A��*

A2S/average_reward_1��'A��@O,       ���E	���z\��A��*

A2S/average_reward_1�q�B�Eu�,       ���E	�&{\��Aʍ*

A2S/average_reward_1��9B#��,       ���E	��D{\��A��*

A2S/average_reward_14�CV�~�,       ���E	j��|\��A��*

A2S/average_reward_1�,D�n x       ��!�	��N�\��A��*i

A2S/klD;

A2S/policy_network_lossA;�

A2S/value_network_lossR�QC

A2S/q_network_loss7?PCsS x,       ���E	?�t�\��A˗*

A2S/average_reward_1�nBq�g,       ���E	����\��A�*

A2S/average_reward_1�޵AQ5�p,       ���E	����\��A��*

A2S/average_reward_1�4�@Sa�,       ���E	g��\��A��*

A2S/average_reward_1~,8D��U�,       ���E	2��\��A��*

A2S/average_reward_1��B���1,       ���E	�@U�\��Aͧ*

A2S/average_reward_1��+DMô�,       ���E	�a]�\��A�*

A2S/average_reward_1m��A����,       ���E	D f�\��A��*

A2S/average_reward_1���A�� ,       ���E	Ֆ��\��AƩ*

A2S/average_reward_1���B�4��,       ���E	=q
�\��A�*

A2S/average_reward_1��PC��-�,       ���E	���\��A��*

A2S/average_reward_1i>�A^��:,       ���E	�G�\��A��*

A2S/average_reward_1�4�A�41,       ���E	sh&�\��A��*

A2S/average_reward_1y#�A)���,       ���E	>�z�\��A��*

A2S/average_reward_1�[MC�{�,       ���E	����\��A�*

A2S/average_reward_1�B'B�Z�s,       ���E	ۥ��\��A��*

A2S/average_reward_1�/BA�|,       ���E	W��\��A��*

A2S/average_reward_14B����,       ���E	?몋\��A̯*

A2S/average_reward_1O��A��g,       ���E	Zr��\��A��*

A2S/average_reward_1�s*B�f�0,       ���E	ղ׋\��A��*

A2S/average_reward_1���Al�,       ���E	�;�\��A�*

A2S/average_reward_1\�]A��,       ���E	5��\��A��*

A2S/average_reward_1��A(?"=,       ���E	�iK�\��Aϲ*

A2S/average_reward_1�8�Bmj�,       ���E	l�y�\��Aϳ*

A2S/average_reward_1#�B)�i8,       ���E	���\��A��*

A2S/average_reward_1<Dї�,       ���E	J� �\��A��*

A2S/average_reward_1T?�C>�S=,       ���E	E�%�\��A��*

A2S/average_reward_1*�D<Ȥ�,       ���E	^9�\��A��*

A2S/average_reward_1g�A�x,       ���E	�bC�\��A��*

A2S/average_reward_1��A��!S,       ���E	l�W�\��A��*

A2S/average_reward_1F�=BYt�,       ���E	b:׏\��A��*

A2S/average_reward_1&�~CKk׾,       ���E	i�#�\��A��*

A2S/average_reward_1�/D",�,       ���E	�s2�\��A��*

A2S/average_reward_1M��A/71�,       ���E	9���\��A��*

A2S/average_reward_1�y)D�t�B,       ���E	����\��A��*

A2S/average_reward_1?��BԮ�H,       ���E	}��\��A��*

A2S/average_reward_1�0D�̠",       ���E	�n�\��A��*

A2S/average_reward_1w�+DYV�K,       ���E	܄�\��A��*

A2S/average_reward_1x�D�.�*,       ���E	��&�\��A��*

A2S/average_reward_1k�A0GrM,       ���E	&�Ɩ\��A��*

A2S/average_reward_1~�Ct���,       ���E	�Ж\��A��*

A2S/average_reward_1��A`��A,       ���E	�2ۖ\��A��*

A2S/average_reward_1a��Ak饽,       ���E	_���\��A��*

A2S/average_reward_1Q��C�3��,       ���E	�$�\��A��*

A2S/average_reward_1˘�C�n,       ���E	��?�\��A��*

A2S/average_reward_1��0B����,       ���E	Yے�\��A��*

A2S/average_reward_1��,Db�O,       ���E	䜙\��A��*

A2S/average_reward_1��	B�*f,       ���E	M�_�\��A��*

A2S/average_reward_1��Cu��|,       ���E	�x�\��Aԇ*

A2S/average_reward_1���BmV��,       ���E	�PV�\��A��*

A2S/average_reward_1�U�C���,       ���E	�z��\��AΎ*

A2S/average_reward_1�C �G�,       ���E	ɿ�\��A��*

A2S/average_reward_1��A{��,       ���E	Y�\�\��A�*

A2S/average_reward_1���C�)),       ���E	N%l�\��A��*

A2S/average_reward_1=�A 8�,       ���E	z/��\��A��*

A2S/average_reward_1���B�#��x       ��!�	n���\��A��*i

A2S/kl��5

A2S/policy_network_loss_�Կ

A2S/value_network_loss�	NC

A2S/q_network_loss�RCV�L�,       ���E	cէ\��A��*

A2S/average_reward_1��$D��C,       ���E	�\�\��A۞*

A2S/average_reward_1^ّC���8,       ���E	[�\��Aҟ*

A2S/average_reward_1Wr�BckJl,       ���E	���\��A��*

A2S/average_reward_1c�B�nO,       ���E	/�C�\��A��*

A2S/average_reward_1C��C7؝,       ���E	�\̪\��A��*

A2S/average_reward_1�s2DO�lp,       ���E	��ժ\��A��*

A2S/average_reward_1=9 B���,       ���E	�n�\��AЫ*

A2S/average_reward_1��A�w�,       ���E	�.B�\��A��*

A2S/average_reward_15Dǵ	,       ���E	��U�\��A�*

A2S/average_reward_1@�cA���v,       ���E	�w��\��Aڻ*

A2S/average_reward_1@8D��p,       ���E	��ݭ\��A��*

A2S/average_reward_1e~SB��_v,       ���E	e��\��A۽*

A2S/average_reward_1�
Cy��P,       ���E	�\��A��*

A2S/average_reward_1�z5D&��s,       ���E	�"�\��A��*

A2S/average_reward_1tt5D�ݹ,       ���E	�Md�\��A��*

A2S/average_reward_1��'CmI͹,       ���E	Qҁ�\��A��*

A2S/average_reward_1�WB1�v$,       ���E	�P�\��A��*

A2S/average_reward_1P��C{���,       ���E	��ǳ\��A��*

A2S/average_reward_1�N3D��į,       ���E	�Kѳ\��A��*

A2S/average_reward_108�AGX
�,       ���E	�N��\��A��*

A2S/average_reward_1SٿCb�T�,       ���E	%w��\��A��*

A2S/average_reward_11��@i�q�,       ���E	r{�\��A��*

A2S/average_reward_19�Bڡ7�,       ���E	B�b�\��A��*

A2S/average_reward_1e6D��m~,       ���E	��{�\��A��*

A2S/average_reward_1�J�Ay�p-,       ���E	����\��A��*

A2S/average_reward_1��'DT,       ���E	&�\��A��*

A2S/average_reward_1̈́!D�LV�,       ���E	n�s�\��Aˀ*

A2S/average_reward_1p5D���&x       ��!�	�I�\��Aˀ*i

A2S/kl��D;

A2S/policy_network_loss�ݿ

A2S/value_network_loss�zC

A2S/q_network_lossi�xC�'�,       ���E	�v�\��A�*

A2S/average_reward_1U`lC��>,       ���E	K��\��A��*

A2S/average_reward_1�.B'ž4,       ���E	g���\��Aȃ*

A2S/average_reward_1ED�A�a�W,       ���E	���\��A�*

A2S/average_reward_1�r�CL��,       ���E	c�.�\��A��*

A2S/average_reward_1>:WBE�oG,       ���E	�k;�\��A·*

A2S/average_reward_1�4�A]<�-,       ���E	~8D�\��A�*

A2S/average_reward_1��A:�d,       ���E	�O�\��A��*

A2S/average_reward_1+B�+�&,       ���E	��W�\��A��*

A2S/average_reward_11��A���,       ���E	�f�\��A��*

A2S/average_reward_1�$�A�?��,       ���E	v:p�\��Aψ*

A2S/average_reward_1��A��,       ���E	!��\��A��*

A2S/average_reward_1H:D����,       ���E	t��\��A�*

A2S/average_reward_1�%BD��,       ���E	�W��\��A��*

A2S/average_reward_1;Y�A��6,       ���E	��\��A��*

A2S/average_reward_1e{C9��,       ���E	|���\��A�*

A2S/average_reward_1WF�C��1,       ���E	�S��\��A��*

A2S/average_reward_1z�B��&�,       ���E	L��\��A�*

A2S/average_reward_1��B���,       ���E	\���\��A��*

A2S/average_reward_1$D�.��,       ���E	B2=�\��A�*

A2S/average_reward_1b3D�F_,       ���E	F̮�\��A��*

A2S/average_reward_1�|C���',       ���E	�x��\��A��*

A2S/average_reward_1 �6DJz/+,       ���E	���\��A��*

A2S/average_reward_1�iBc��b,       ���E	�f�\��A��*

A2S/average_reward_1&�6Dǽ�
,       ���E	�2��\��Aɸ*

A2S/average_reward_1��vC���",       ���E	jt<�\��A��*

A2S/average_reward_11j9Dw5܉,       ���E	��M�\��A��*

A2S/average_reward_1C�5B��t,       ���E	\�[�\��A��*

A2S/average_reward_1��B�.�.,       ���E	.�h�\��A��*

A2S/average_reward_1e�BѰ~,       ���E	Qsv�\��A��*

A2S/average_reward_1բ�AB[q�,       ���E	�0��\��A��*

A2S/average_reward_1*AAB���,       ���E	���\��A��*

A2S/average_reward_1��C+g�g,       ���E	����\��A��*

A2S/average_reward_1w@KC���_,       ���E	D�<�\��A��*

A2S/average_reward_1��C�+��,       ���E	�u��\��A��*

A2S/average_reward_1W)�C��u,       ���E	�A�\��A��*

A2S/average_reward_1֎(Du`��,       ���E	�X�\��A��*

A2S/average_reward_1�'�C�_0�,       ���E	9��\��A��*

A2S/average_reward_1�H;B�H��,       ���E	�8�\��A��*

A2S/average_reward_16-DGП,       ���E	XH�\��A��*

A2S/average_reward_1.�BT�N,       ���E	�ZR�\��A��*

A2S/average_reward_1v`�A�N�L,       ���E	���\��A��*

A2S/average_reward_1�DM�tD,       ���E	���\��A��*

A2S/average_reward_1�r+DH�e(,       ���E	����\��A��*

A2S/average_reward_1r��ASB%�,       ���E	��r�\��A��*

A2S/average_reward_1c�+Do"E�,       ���E	����\��A��*

A2S/average_reward_1T�C�O5 ,       ���E	�P�\��A��*

A2S/average_reward_1>U*C� ;�,       ���E	����\��A܂*

A2S/average_reward_1N��C���,       ���E	0=G�\��AĊ*

A2S/average_reward_1[�+D��9�,       ���E	�lK�\��A��*

A2S/average_reward_1x� D��8,       ���E	��\��A��*

A2S/average_reward_13y/D�,       ���E	{���\��AØ*

A2S/average_reward_1��A�},       ���E	ʬF�\��A��*

A2S/average_reward_1e*D�$�P,       ���E	����\��A��*

A2S/average_reward_1+�+D���,       ���E	����\��A��*

A2S/average_reward_1-l�A��~�,       ���E	x�R�\��Aʫ*

A2S/average_reward_1�A�Cd��,       ���E	G���\��A��*

A2S/average_reward_1�TC��k�,       ���E	��%�\��A�*

A2S/average_reward_1Q�C�<9 ,       ���E	!�1�\��A��*

A2S/average_reward_1�2nA��,       ���E	��v�\��A�*

A2S/average_reward_1��C6�Z=,       ���E	����\��A��*

A2S/average_reward_1/c!D
�Q�,       ���E	J3��\��A��*

A2S/average_reward_1�)B�"#�,       ���E	����\��A�*

A2S/average_reward_14SBI�,       ���E	�J��\��A��*

A2S/average_reward_1�sA���,       ���E	�g4�\��A¼*

A2S/average_reward_1�pC�5],       ���E	 ���\��A��*

A2S/average_reward_1�x�CvL��,       ���E	�G��\��A��*

A2S/average_reward_1@B�C$-�b,       ���E	v��\��A��*

A2S/average_reward_1V-D	��,       ���E	1���\��A��*

A2S/average_reward_1��Bs�ů,       ���E	v��\��A��*

A2S/average_reward_1Q�BWOB},       ���E	_A��\��A��*

A2S/average_reward_1�v�C�um,       ���E	h���\��A��*

A2S/average_reward_1��B��I�,       ���E	��\��A��*

A2S/average_reward_1���A|��,       ���E	(-��\��A��*

A2S/average_reward_1�aB���,       ���E	���\��A��*

A2S/average_reward_1�{{A��=�,       ���E	�\��A��*

A2S/average_reward_1m��B�,       ���E	�$�\��A��*

A2S/average_reward_1��BZ ��,       ���E	@�2�\��A��*

A2S/average_reward_1��AӸ�,       ���E	�B�\��A��*

A2S/average_reward_1���A�e�,       ���E	ƩH�\��A��*

A2S/average_reward_1j��A��zJ,       ���E	�?V�\��A��*

A2S/average_reward_1��B{�oY,       ���E	�A`�\��A��*

A2S/average_reward_1���A��%�,       ���E	����\��A��*

A2S/average_reward_1�7C%�PL,       ���E	����\��A��*

A2S/average_reward_1?�D���,       ���E	�?�\��A��*

A2S/average_reward_1�)D�fG,       ���E	�WQ�\��A��*

A2S/average_reward_1�~B�<�,       ���E	a�^�\��A��*

A2S/average_reward_1'��A{� �x       ��!�	���\��A��*i

A2S/kl�ʩ:

A2S/policy_network_lossR��

A2S/value_network_loss6�RC

A2S/q_network_loss��UC	h�$,       ���E	����\��A��*

A2S/average_reward_1ea�A���c,       ���E	��X�\��A��*

A2S/average_reward_1�-�C�1Ê,       ���E	�i�\��A��*

A2S/average_reward_1;'NBdT,       ���E	%���\��A��*

A2S/average_reward_1E��B:�tL,       ���E	Um��\��A��*

A2S/average_reward_1ZM�Ab�H,       ���E	³��\��A��*

A2S/average_reward_1��B�@%�,       ���E	l߻�\��A��*

A2S/average_reward_1��BI�R ,       ���E	w�M�\��A��*

A2S/average_reward_1���C�"x�,       ���E	2>��\��A��*

A2S/average_reward_1�J�C�J,       ���E	�x��\��A��*

A2S/average_reward_1�60B�7S�,       ���E	���\��A��*

A2S/average_reward_1�D���,       ���E	'�\��A��*

A2S/average_reward_1U�B}f�K,       ���E	55E�\��A��*

A2S/average_reward_1��B�.,       ���E	�sR�\��A��*

A2S/average_reward_1�p�Aϰk�,       ���E	]g��\��A��*

A2S/average_reward_1��jC\�I ,       ���E	��	�\��A��*

A2S/average_reward_1Z15D�q�,       ���E	�S�\��A�*

A2S/average_reward_1�v3D�Z�,       ���E	�
�\��A��*

A2S/average_reward_1Z$�C��,       ���E	���\��AԒ*

A2S/average_reward_1τ�A�/�u,       ���E	VnF�\��A��*

A2S/average_reward_1���B��:�,       ���E	��T�\��A��*

A2S/average_reward_1��-BM�F�,       ���E	XB��\��A��*

A2S/average_reward_1��2Dl�/�,       ���E	����\��A��*

A2S/average_reward_1��@�1�,       ���E	:U' ]��A��*

A2S/average_reward_1�3Cţ�,       ���E	d� ]��A��*

A2S/average_reward_1�a3Cʼ�,       ���E	G
� ]��A��*

A2S/average_reward_1�v3Cd��},       ���E	U��]��A��*

A2S/average_reward_1�XD1��;,       ���E	�G]��A��*

A2S/average_reward_1s�hB��,       ���E	�:?]��A�*

A2S/average_reward_1T9D,I�m,       ���E	��D]��A�*

A2S/average_reward_1Æ�A|���x       ��!�	��<]��A�*i

A2S/kl+dh;

A2S/policy_network_lossɌ
�

A2S/value_network_loss��lC

A2S/q_network_loss5kC)��8,       ���E	�S�]��Aҷ*

A2S/average_reward_1^U"DMPj�,       ���E	S��]��A��*

A2S/average_reward_1m{[B��:5,       ���E	Gx�]��A�*

A2S/average_reward_1��Cl�,       ���E	�1�]��AӺ*

A2S/average_reward_1�;�BgO\f,       ���E	a�]��A��*

A2S/average_reward_1?�CBۧ,       ���E	�1�]��A��*

A2S/average_reward_1�IB�P��,       ���E	��]��A�*

A2S/average_reward_1��Cȳl,       ���E	;��]��A��*

A2S/average_reward_1v��A*e�/,       ���E	�]��A��*

A2S/average_reward_1��B&�/,       ���E	�N�]��A��*

A2S/average_reward_1�~�C��e�,       ���E	G�]��A��*

A2S/average_reward_1/eB���,       ���E	q�]��A��*

A2S/average_reward_1=�Bܓf�,       ���E	�,�]��A��*

A2S/average_reward_1��Bi��y,       ���E	$�m]��A��*

A2S/average_reward_1���C.6�a,       ���E	�1]��A��*

A2S/average_reward_1�p�C��r�,       ���E	���]��A��*

A2S/average_reward_1�x�Ce2��,       ���E	�o�]��A��*

A2S/average_reward_1�|�A{�~�,       ���E	�]��A��*

A2S/average_reward_1۟B����,       ���E	=%]��A��*

A2S/average_reward_1WF�A|�ά,       ���E	{jR]��A��*

A2S/average_reward_1*�B�H$�,       ���E	_�_]��A��*

A2S/average_reward_1�Y�AtT}�,       ���E	zCj]��A��*

A2S/average_reward_1�i�AA��|,       ���E	��]��A��*

A2S/average_reward_1���B[`�f,       ���E	�@�]��A��*

A2S/average_reward_1�M�AC��,       ���E	��]��A��*

A2S/average_reward_1�(�A����,       ���E	v�]��A��*

A2S/average_reward_1�=DBJ���,       ���E	[�]��A��*

A2S/average_reward_1:�A�#S,       ���E	��]��A��*

A2S/average_reward_1�"�A8Z:�,       ���E	x��]��A��*

A2S/average_reward_1�,CLە�,       ���E	���]��A��*

A2S/average_reward_1<��AyD=x       ��!�	O��]��A��*i

A2S/kl��;

A2S/policy_network_loss��Q�

A2S/value_network_loss��C

A2S/q_network_loss�B�CA<�,       ���E	X��]��A��*

A2S/average_reward_1f�B�׺,       ���E	���]��A��*

A2S/average_reward_1���A�^3�,       ���E	�mX]��A��*

A2S/average_reward_1�vCg��,       ���E	ioo]��A��*

A2S/average_reward_1��A~�P,       ���E	S��]��A��*

A2S/average_reward_1w~�B����,       ���E	�o�]��A��*

A2S/average_reward_1��A�Dxo,       ���E	��]��A��*

A2S/average_reward_1�_�B�S)i,       ���E	��h]��A��*

A2S/average_reward_1�b�C��G�,       ���E	�W�]��A��*

A2S/average_reward_1Ze=C�Vi�,       ���E	���]��A��*

A2S/average_reward_18�B�nM�,       ���E	џ�]��A��*

A2S/average_reward_1I,�Bf��,       ���E	�o�]��A��*

A2S/average_reward_1':'B�s~,       ���E	01]��A��*

A2S/average_reward_1|��B]���,       ���E	��>]��A��*

A2S/average_reward_1M�AS�w,       ���E	�/Z]��A��*

A2S/average_reward_1��nB�6�,       ���E	�e]��A��*

A2S/average_reward_1���A�o,       ���E	��]��A��*

A2S/average_reward_1{D6C$��_,       ���E	���]��A��*

A2S/average_reward_1�v�B�*��,       ���E	���]��A��*

A2S/average_reward_1)G�Ay�c�,       ���E	�g$ ]��A��*

A2S/average_reward_1�2CB`w�,       ���E	��, ]��A��*

A2S/average_reward_1)(�AWM�,       ���E	��V ]��A��*

A2S/average_reward_1W*�B��0',       ���E	G�c ]��A��*

A2S/average_reward_12�B1[,       ���E	�k!]��A��*

A2S/average_reward_1pn�C><{�,       ���E	��!]��A��*

A2S/average_reward_1a0�A#y,       ���E	��!]��A��*

A2S/average_reward_1��B�a%�,       ���E	+F7!]��A��*

A2S/average_reward_1�[�B{�z�,       ���E	M,H!]��A��*

A2S/average_reward_15�tB�e#�,       ���E	i.s!]��A��*

A2S/average_reward_1�XC���,       ���E	�6�!]��A��*

A2S/average_reward_1��XB}.�,       ���E	B<�!]��A��*

A2S/average_reward_1�I�A(ђ ,       ���E	���"]��A��*

A2S/average_reward_1U3+Di�k�,       ���E	^��"]��A��*

A2S/average_reward_1�F�Aϣ�,       ���E	�Y�#]��A��*

A2S/average_reward_1&T�C~]��,       ���E	x�#]��A��*

A2S/average_reward_1�XBJ��K,       ���E	���#]��A��*

A2S/average_reward_1��LB�C�,       ���E	��T%]��A��*

A2S/average_reward_1-D�T;N,       ���E	f��&]��A�*

A2S/average_reward_1��,D�x�;,       ���E	��&]��A��*

A2S/average_reward_1��BgE�C,       ���E	+P']��A��*

A2S/average_reward_1��C
qψ,       ���E	��(]��Aܚ*

A2S/average_reward_1A:,D�N!1,       ���E	�"�)]��A��*

A2S/average_reward_1���Ce!�,,       ���E	^�)]��A *

A2S/average_reward_1��A���,       ���E	V��)]��A��*

A2S/average_reward_1��A�׆:,       ���E	���)]��A��*

A2S/average_reward_1��YB
�E�,       ���E	j�)]��Aѡ*

A2S/average_reward_1/��A���$,       ���E	��)]��A��*

A2S/average_reward_1�̡@`�,       ���E	x��*]��A��*

A2S/average_reward_1�j�C�@r�,       ���E	I�+]��A��*

A2S/average_reward_1P��Cڃ�e,       ���E	��+]��A��*

A2S/average_reward_1��B�&�*,       ���E	-��+]��A��*

A2S/average_reward_1R�@ da�,       ���E	],]��A̮*

A2S/average_reward_1�e�A̵Ϭ,       ���E	��:,]��A�*

A2S/average_reward_1^S�B��ʊ,       ���E	�9�,]��A��*

A2S/average_reward_1q�NC+=�s,       ���E	sM�,]��A��*

A2S/average_reward_1?BD���,       ���E	�x�,]��Aݲ*

A2S/average_reward_1�<�A���,       ���E	ϊ.]��Aź*

A2S/average_reward_1o�-D>�m�,       ���E	d'0.]��A��*

A2S/average_reward_1;�,B�IO<,       ���E	�)�/]��A��*

A2S/average_reward_1A�*Dws��,       ���E	\�k0]��A��*

A2S/average_reward_1��CaLea,       ���E	��t0]��A��*

A2S/average_reward_1���A¬��,       ���E	x�Y1]��A��*

A2S/average_reward_1�,�C`5�_,       ���E	�2]��A��*

A2S/average_reward_1��+D��t�,       ���E	O�2]��A��*

A2S/average_reward_1Bn�`,       ���E	��2]��A��*

A2S/average_reward_1� �A���,       ���E	"ث2]��A��*

A2S/average_reward_1[]�A$b�$,       ���E	%4]��A��*

A2S/average_reward_1wq-D1i��,       ���E	�+4]��A��*

A2S/average_reward_1#)�A	z�,       ���E	\h�4]��A��*

A2S/average_reward_18��C%��,       ���E	��5]��A��*

A2S/average_reward_1[�cBb��,       ���E	��X6]��A��*

A2S/average_reward_1<�.D&,I,       ���E	�l�6]��A��*

A2S/average_reward_1�G|C�� e,       ���E	`#�6]��A��*

A2S/average_reward_1�B�QS,       ���E	���6]��A��*

A2S/average_reward_1��Bi�h�,       ���E	�P7]��A��*

A2S/average_reward_1�?�B-=�,       ���E	Zxs8]��A��*

A2S/average_reward_1��,Dn.�s,       ���E	d){8]��A��*

A2S/average_reward_1��A�U�,       ���E	�h�8]��A��*

A2S/average_reward_1y-B���M,       ���E	D��8]��A��*

A2S/average_reward_1�V�A�Jo,       ���E	.�8]��A��*

A2S/average_reward_1��A�=F+,       ���E	���8]��A��*

A2S/average_reward_1�P�A;�(�,       ���E	_a�9]��A��*

A2S/average_reward_1���Ca���,       ���E	
(�9]��A��*

A2S/average_reward_1�0B9�,       ���E	ok�9]��A��*

A2S/average_reward_1d"�A�2,       ���E	��9]��A��*

A2S/average_reward_1�B��7,       ���E	!%;]��Aׅ*

A2S/average_reward_1k_,D��`,       ���E	{&<;]��A��*

A2S/average_reward_1k]'BE�HM,       ���E	��H;]��AƆ*

A2S/average_reward_1�d�A�Q�,       ���E	(��;]��A��*

A2S/average_reward_1�C���,       ���E	�3#<]��A׋*

A2S/average_reward_1I��B<Jkz,       ���E	�)=<]��A��*

A2S/average_reward_1`NABM��p,       ���E	o�<]��Aލ*

A2S/average_reward_1FuC�U�,       ���E	�{=]��A��*

A2S/average_reward_1 8�CwmV�x       ��!�	>��F]��A��*i

A2S/kl�B�<

A2S/policy_network_loss��߿

A2S/value_network_loss~!]C

A2S/q_network_lossg�`C�Q��,       ���E	���F]��A��*

A2S/average_reward_1��A��ˀ,       ���E	�1G]��Aב*

A2S/average_reward_1���A	]�,       ���E	�)G]��A��*

A2S/average_reward_1w�3A����,       ���E	�@3G]��AÒ*

A2S/average_reward_1" =Bb�Ft,       ���E	ZWCG]��A�*

A2S/average_reward_1���A����,       ���E	��PG]��A��*

A2S/average_reward_1�/B��V�,       ���E	T�[G]��A��*

A2S/average_reward_1�eB���>,       ���E	R�H]��A��*

A2S/average_reward_1��9D�#�,       ���E	���I]��A��*

A2S/average_reward_1$29DuD�4,       ���E	�1DJ]��A��*

A2S/average_reward_1_�<C�s��,       ���E	��J]��A��*

A2S/average_reward_1�xCu�v-,       ���E	���J]��A��*

A2S/average_reward_1���A밖�,       ���E	���K]��A��*

A2S/average_reward_1��2D��I�,       ���E	�[�K]��A��*

A2S/average_reward_1���Aچ�,       ���E	�rBL]��AƱ*

A2S/average_reward_1
,C��,       ���E	�{JL]��A�*

A2S/average_reward_1I�B3��l,       ���E	qRL]��A��*

A2S/average_reward_1@��A�Y�,       ���E	B�,M]��A��*

A2S/average_reward_1@*�CJ���,       ���E	6<M]��A��*

A2S/average_reward_1W�B�4�d,       ���E	��`M]��A��*

A2S/average_reward_1���B֚.C,       ���E	���N]��A��*

A2S/average_reward_1��(Dd`�H,       ���E	�ޖN]��A��*

A2S/average_reward_1�A���,       ���E	ydO]��A��*

A2S/average_reward_1���C���,       ���E	R��O]��A��*

A2S/average_reward_1�0/CRO��,       ���E	���P]��A��*

A2S/average_reward_19�/Dt���,       ���E	�>Q]��A��*

A2S/average_reward_1��!C�r�,       ���E	�"R]��A��*

A2S/average_reward_1{��C��6�,       ���E	��.R]��A��*

A2S/average_reward_1J��@v��.,       ���E	I2�R]��A��*

A2S/average_reward_1.�SC�)L�,       ���E	��R]��A��*

A2S/average_reward_1>H:B�c��,       ���E	��R]��A��*

A2S/average_reward_1���B.�G�,       ���E	M�R]��A��*

A2S/average_reward_1��B���x       ��!�	�d\]��A��*i

A2S/kl�89

A2S/policy_network_loss�
�

A2S/value_network_losst߁C

A2S/q_network_loss�r�C���,       ���E	�o\]��A��*

A2S/average_reward_1�\�A|�E�,       ���E	6��\]��A��*

A2S/average_reward_10.B�i�],       ���E	.��\]��A��*

A2S/average_reward_1�WCf��,       ���E	n�5]]��A��*

A2S/average_reward_1K�C��?S,       ���E	�5@]]��A��*

A2S/average_reward_1������y,       ���E	M�L]]��A��*

A2S/average_reward_1�}B]�7,       ���E	�vU]]��A��*

A2S/average_reward_1J#�Ay�!,       ���E	ⷑ]]��A��*

A2S/average_reward_1�d�B�,       ���E	�~�]]��A��*

A2S/average_reward_12��B��н,       ���E	���]]��A��*

A2S/average_reward_1�n�Ah�u�,       ���E	-29^]��A��*

A2S/average_reward_1<̃C�g��,       ���E	�gB^]��A��*

A2S/average_reward_1݉B��^T,       ���E	$��^]��A��*

A2S/average_reward_1�{gC��,       ���E	���^]��A��*

A2S/average_reward_1̠"C=��4,       ���E	���^]��A��*

A2S/average_reward_1�Q�Aʷ��,       ���E	фn_]��A��*

A2S/average_reward_1]C�C�w�,       ���E	^�_]��A��*

A2S/average_reward_1^� Cڜ�,       ���E	QǮ_]��A��*

A2S/average_reward_1�:BsX��,       ���E	#��_]��A��*

A2S/average_reward_1 �KC����,       ���E	d\�`]��A��*

A2S/average_reward_1��Cs���,       ���E		��`]��A��*

A2S/average_reward_1^��B�(�,       ���E	_i�`]��A��*

A2S/average_reward_1ez�A���,       ���E	�dDa]��A��*

A2S/average_reward_1-�[C�n�2,       ���E	��Xa]��A��*

A2S/average_reward_1s�ZB�:�,       ���E	1H�a]��A��*

A2S/average_reward_1�+YC?w�,       ���E	5�a]��A��*

A2S/average_reward_1]UAb��K,       ���E	���a]��A��*

A2S/average_reward_1���B�b�<,       ���E	yZjb]��A��*

A2S/average_reward_1s��C�2��,       ���E	�R�b]��A��*

A2S/average_reward_12Cڲ��,       ���E	�x�b]��A��*

A2S/average_reward_1�B�f&,       ���E	Б�b]��A��*

A2S/average_reward_1��
C4�~,       ���E	zXc]��A��*

A2S/average_reward_1C{L,       ���E	~�c]��A�*

A2S/average_reward_1�cB�H +,       ���E	�?Xd]��Aǈ*

A2S/average_reward_1�Q-D4ob�,       ���E	���d]��A��*

A2S/average_reward_1f�ZC�ǋ0,       ���E	ş�d]��A֋*

A2S/average_reward_1�b
BY��%,       ���E	4��d]��A��*

A2S/average_reward_1��B�),       ���E	e�e]��A��*

A2S/average_reward_1e-�A;P�z,       ���E	�De]��A��*

A2S/average_reward_1���B|CY�,       ���E	�IWe]��A��*

A2S/average_reward_1�QHA�F��,       ���E	���e]��A��*

A2S/average_reward_1R��C�p�-,       ���E	�[f]��A��*

A2S/average_reward_1���A���,       ���E	JMf]��AΒ*

A2S/average_reward_1��Aba�a,       ���E	�N�f]��A��*

A2S/average_reward_1�=�C(a��,       ���E	�_h]��AÞ*

A2S/average_reward_1x D�V�,       ���E	�$Bi]��A��*

A2S/average_reward_1�)D��,       ���E	Hsj]��A��*

A2S/average_reward_1�4+D߽;,       ���E	���j]��A��*

A2S/average_reward_1���B�H�,       ���E	���j]��AЯ*

A2S/average_reward_1�n�AL\O,       ���E	O"�j]��A�*

A2S/average_reward_1YHR@�ø,       ���E	S� l]��Aڷ*

A2S/average_reward_1,D�'�,       ���E	Nj~l]��A��*

A2S/average_reward_1���C�z�R,       ���E	zH�m]��A��*

A2S/average_reward_1ʺ Dϴ�,       ���E	ӂ%o]��A��*

A2S/average_reward_1Z�Dբ1�,       ���E	#8$p]��A��*

A2S/average_reward_1*�Cv1c�,       ���E	ȓ0p]��A��*

A2S/average_reward_1QGB�߻,       ���E	�Y>p]��A��*

A2S/average_reward_1���A�g	b,       ���E	��Pp]��A��*

A2S/average_reward_1��B#��,       ���E	��p]��A��*

A2S/average_reward_1��C���,       ���E	`^�q]��A��*

A2S/average_reward_1�%-D�w`$,       ���E	�Ds]��A��*

A2S/average_reward_1�)D�2h�,       ���E	p6xs]��A��*

A2S/average_reward_1���B�v��,       ���E	��s]��A��*

A2S/average_reward_1�6_C���,       ���E	)Kt]��A��*

A2S/average_reward_1��CC��~g,       ���E	��dt]��A��*

A2S/average_reward_1W1�B���,       ���E	�Kst]��A��*

A2S/average_reward_1҃�@=i:z,       ���E	T�}t]��A��*

A2S/average_reward_1��A�)lM,       ���E	�C�t]��A��*

A2S/average_reward_1h|gB�X��,       ���E	՛�u]��A��*

A2S/average_reward_1�*D��C\,       ���E	�y"v]��A��*

A2S/average_reward_1�C0��,       ���E	��2v]��A��*

A2S/average_reward_1vBM��,       ���E	,\Rw]��A��*

A2S/average_reward_1Y/DVw"F,       ���E	�9lw]��A��*

A2S/average_reward_1�~OB��2�,       ���E	m�w]��A��*

A2S/average_reward_1��B�},       ���E	Aڟw]��A��*

A2S/average_reward_1!�A�f�y,       ���E	mZ�w]��A��*

A2S/average_reward_1E�B�$ж,       ���E	=�y]��A��*

A2S/average_reward_1�*-D��Q�,       ���E	�&y]��A��*

A2S/average_reward_1�Y�A"51=,       ���E	`�yy]��A��*

A2S/average_reward_1}�C���f,       ���E	�J�y]��Aل*

A2S/average_reward_1bm�B	��,       ���E	�s�y]��A�*

A2S/average_reward_1���A�MN�,       ���E	6ɫy]��A��*

A2S/average_reward_1�ϻAv�υ,       ���E	��z]��AǇ*

A2S/average_reward_1B�`CW{D�,       ���E	Y�W{]��A��*

A2S/average_reward_1��+DH�5,       ���E	��_{]��Aȏ*

A2S/average_reward_1/��A'>�,       ���E	���{]��A��*

A2S/average_reward_1UFMCĤ8S,       ���E	/@�{]��A��*

A2S/average_reward_1`I.B��,       ���E	G�}]��A��*

A2S/average_reward_1�p.D�th,       ���E	z0x~]��A��*

A2S/average_reward_1csD5�,       ���E	��~]��Aá*

A2S/average_reward_1�VB��,       ���E	$�~]��A��*

A2S/average_reward_1ɋ�B��U�,       ���E	O��~]��A��*

A2S/average_reward_1�B�ޘ�,       ���E	�9�]��A��*

A2S/average_reward_1f *D�J\,       ���E	Y�a�]��A��*

A2S/average_reward_1��Dpd��,       ���E	�rk�]��A��*

A2S/average_reward_10��A���,       ���E	y8u�]��A��*

A2S/average_reward_1p�AHȓ|,       ���E	�V��]��A�*

A2S/average_reward_1BLB+�!,       ���E	��b�]��A�*

A2S/average_reward_1���C.9�1,       ���E	nUp�]��A��*

A2S/average_reward_11��AHUI_x       ��!�	��g�]��A��*i

A2S/kl3�&9

A2S/policy_network_loss���

A2S/value_network_lossG@C

A2S/q_network_lossҏBC�1k,       ���E	�K،]��Aغ*

A2S/average_reward_1�QzC���,       ���E	>v�]��A��*

A2S/average_reward_1%�B�. ,       ���E	v�>�]��A��*

A2S/average_reward_1M32D�,9,       ���E	mye�]��A��*

A2S/average_reward_1U	�B�zu4,       ���E	���]��A��*

A2S/average_reward_1�t�C6�S_,       ���E	��+�]��A��*

A2S/average_reward_1x?�B��%�,       ���E	���]��A��*

A2S/average_reward_1^�8D��a,       ���E	�d	�]��A��*

A2S/average_reward_1��C�ȁ�,       ���E	Wq�]��A��*

A2S/average_reward_1��SC���,       ���E	�s��]��A��*

A2S/average_reward_1A%C`Js�,       ���E	��ȑ]��A��*

A2S/average_reward_1���B��C",       ���E	�a'�]��A��*

A2S/average_reward_1�X]C��ܸ,       ���E	�Ē]��A��*

A2S/average_reward_1��C�\k�,       ���E	KH�]��A��*

A2S/average_reward_1�,�B��M�,       ���E	��]��A��*

A2S/average_reward_1�DЁ�,       ���E	SS�]��A��*

A2S/average_reward_1"+B�eO�,       ���E	7T��]��A��*

A2S/average_reward_1�l�A��eX,       ���E	�g�]��A��*

A2S/average_reward_1Z)xBΓ 7,       ���E	���]��A��*

A2S/average_reward_1^9�C�;�,       ���E	^U�]��A��*

A2S/average_reward_1L�1C�f�,       ���E	4W#�]��A��*

A2S/average_reward_1��B�n��,       ���E	P�,�]��A��*

A2S/average_reward_1э�A{�,       ���E	��6�]��A��*

A2S/average_reward_1ɐ[At���,       ���E	+[=�]��A��*

A2S/average_reward_1\ }A*%�,       ���E	�_�]��A��*

A2S/average_reward_1��HBR�E,       ���E	F�g�]��A��*

A2S/average_reward_1�n�A�&m,       ���E	�ľ�]��A��*

A2S/average_reward_1k�SC�^�i,       ���E	�^ɕ]��A��*

A2S/average_reward_1�bBSV��,       ���E	t>�]��A��*

A2S/average_reward_1�G�B���,       ���E	wzE�]��A��*

A2S/average_reward_1��B����,       ���E	Y�]��A��*

A2S/average_reward_1Bn-B�,       ���E	cb�]��A��*

A2S/average_reward_13��A���;,       ���E	ǰj�]��A��*

A2S/average_reward_19��A�q,       ���E	�`u�]��A��*

A2S/average_reward_1�Aރ{*,       ���E	��]��A��*

A2S/average_reward_1�~�A�D3,       ���E	�p��]��A��*

A2S/average_reward_1FD�A�X��,       ���E	@0��]��A��*

A2S/average_reward_1D@�?�j(�,       ���E	�t��]��A��*

A2S/average_reward_1m�.C��*,       ���E	��ڗ]��A��*

A2S/average_reward_1���C6� ,       ���E	ǔ�]��A��*

A2S/average_reward_1�XC瘚B,       ���E	�'�]��A��*

A2S/average_reward_1�;�A����,       ���E	!F/�]��A��*

A2S/average_reward_1{AkL�,       ���E	)�`�]��A��*

A2S/average_reward_1v(�B
��7,       ���E	(-z�]��A��*

A2S/average_reward_1��Av�wB,       ���E	]��A��*

A2S/average_reward_1P=Dഄ,       ���E	��]��A�*

A2S/average_reward_1��^ClO�),       ���E	��.�]��A��*

A2S/average_reward_1 �$DU�hZ,       ���E	�z:�]��Aˏ*

A2S/average_reward_1ÿA�F�,       ���E	�~A�]��A�*

A2S/average_reward_1S��A���,       ���E	p�O�]��A��*

A2S/average_reward_1nRB>�1�,       ���E	QLi�]��A�*

A2S/average_reward_1�B�bF�,       ���E	�|�]��A��*

A2S/average_reward_1L~BEwYV,       ���E	p,��]��Aё*

A2S/average_reward_1h<�A�+¿,       ���E	"���]��A��*

A2S/average_reward_1/�OB%6��,       ���E	?���]��A��*

A2S/average_reward_1�_:C�c!,       ���E	 �T�]��A��*

A2S/average_reward_1�� D���,       ���E	��_�]��A�*

A2S/average_reward_1 �BI�s,       ���E	ҙɞ]��Aɣ*

A2S/average_reward_1�}0Dя��,       ���E	�C�]��A��*

A2S/average_reward_1v^C�4�,       ���E	�:O�]��A��*

A2S/average_reward_1"B�@�,       ���E	y���]��A��*

A2S/average_reward_1%�-D��-,       ���E	Z�à]��A��*

A2S/average_reward_1S�A���,       ���E	.�Ҡ]��A߮*

A2S/average_reward_1�K�A|X�,       ���E	j�]��A�*

A2S/average_reward_1y��B�}��,       ���E	�AH�]��A��*

A2S/average_reward_1&��BYr�,       ���E	o[�]��A��*

A2S/average_reward_1:�	D��-�,       ���E	�>j�]��Aӷ*

A2S/average_reward_1"��@���,       ���E	eH��]��A��*

A2S/average_reward_1��,D[���,       ���E	~���]��A��*

A2S/average_reward_1�s)Dj��,       ���E	0��]��A��*

A2S/average_reward_1�$�A<&��,       ���E	;��]��A��*

A2S/average_reward_1E�A7�}�,       ���E	�O�]��A��*

A2S/average_reward_1f*Di�S,       ���E	�]p�]��A��*

A2S/average_reward_1F��B��� ,       ���E	���]��A��*

A2S/average_reward_16��C��{,       ���E	m���]��A��*

A2S/average_reward_1*�)B �,       ���E	.��]��A��*

A2S/average_reward_1 A�R��,       ���E	�w �]��A��*

A2S/average_reward_1��B�gP,       ���E	��F�]��A��*

A2S/average_reward_1� �BV鍗,       ���E	@�l�]��A��*

A2S/average_reward_1uB�ۿ,       ���E	�Y��]��A��*

A2S/average_reward_1kR�B�l�O,       ���E	�ߓ�]��A��*

A2S/average_reward_1��A
�,       ���E	��ɧ]��A��*

A2S/average_reward_1[�B>�7{,       ���E	$/�]��A��*

A2S/average_reward_1�QB�ϘW,       ���E	�;��]��A��*

A2S/average_reward_1|_�Aԟe�,       ���E	�{.�]��A��*

A2S/average_reward_1=�B*�R�,       ���E	K�Z�]��A��*

A2S/average_reward_1�z,D�4H,       ���E	���]��A��*

A2S/average_reward_1�ɪC�f��,       ���E	�Y��]��A��*

A2S/average_reward_1q�nC�� �,       ���E	�ߺ�]��A��*

A2S/average_reward_1o��B`�H�,       ���E	�X�]��A��*

A2S/average_reward_1�+-D�޻�,       ���E	�|�]��A��*

A2S/average_reward_1n��C�v�,       ���E	���]��A��*

A2S/average_reward_1�̓A�%m�,       ���E	La$�]��A��*

A2S/average_reward_1{U�A��J�,       ���E	�4�]��A��*

A2S/average_reward_1x��A���,       ���E	M��]��A��*

A2S/average_reward_1�B+Dz�V�,       ���E	2L!�]��A��*

A2S/average_reward_1<-D�ⲡ,       ���E	T;�]��Aۈ*

A2S/average_reward_1��jB4ȕk,       ���E	o�S�]��AŎ*

A2S/average_reward_1mD1ZB,       ���E	�\�]��Aގ*

A2S/average_reward_1nW�A���T,       ���E	Z�k�]��A��*

A2S/average_reward_1L�B��v,       ���E	�Iy�]��A��*

A2S/average_reward_1?�B �m,       ���E	���]��Aʏ*

A2S/average_reward_1C �AV�,       ���E	V��]��A��*

A2S/average_reward_1ʶ�@FM�!,       ���E	�}��]��A��*

A2S/average_reward_1 �=A�U�,       ���E	����]��A��*

A2S/average_reward_1�s D?��x       ��!�	<���]��A��*i

A2S/klN~<

A2S/policy_network_loss�0�

A2S/value_network_loss�kUC

A2S/q_network_lossYd[C�q�,       ���E	��]��AЖ*

A2S/average_reward_1�H&Bu�7�,       ���E	�"�]��Aؗ*

A2S/average_reward_1�s�B*r,       ���E	n���]��A��*

A2S/average_reward_1�ْB�5�,       ���E	����]��A��*

A2S/average_reward_1���A�"��,       ���E	���]��AȘ*

A2S/average_reward_1�Y�Ar���,       ���E	86�]��A٘*

A2S/average_reward_1Ր�A��b[,       ���E	9;�]��A�*

A2S/average_reward_1j�A ���,       ���E	��5�]��Aљ*

A2S/average_reward_1���B-�M�,       ���E	��G�]��A��*

A2S/average_reward_1v�Be���,       ���E	>�U�]��A��*

A2S/average_reward_1�tKB	�,       ���E	�=_�]��A֚*

A2S/average_reward_1	�B�s&�,       ���E	2�l�]��A��*

A2S/average_reward_1�"B��,       ���E	�"��]��A��*

A2S/average_reward_1w6�B0��,       ���E	�å�]��A��*

A2S/average_reward_1���A���,       ���E	^ǽ]��A��*

A2S/average_reward_1�k�B�?n},       ���E	��]��A�*

A2S/average_reward_1���B��K�,       ���E	����]��A��*

A2S/average_reward_1s��A��jA,       ���E	����]��A��*

A2S/average_reward_1��Af���,       ���E	C%�]��A��*

A2S/average_reward_1���B9�D�,       ���E	��B�]��Aҟ*

A2S/average_reward_1>��B��,       ���E	�N�]��A��*

A2S/average_reward_1�B�c,       ���E	�Z�]��A��*

A2S/average_reward_1���A6�G�,       ���E	]�|�]��A��*

A2S/average_reward_1�^�B�^�,       ���E	%�]��A��*

A2S/average_reward_1���B���,,       ���E	{�¾]��A��*

A2S/average_reward_1>�.B�]-�,       ���E	�۾]��A��*

A2S/average_reward_1�B ���,       ���E	,�]��A��*

A2S/average_reward_1��AA?7,       ���E	���]��A��*

A2S/average_reward_1m��A�z.�,       ���E	��]��A��*

A2S/average_reward_1��nB��:,       ���E	�C�]��Aʥ*

A2S/average_reward_1�D2C|��?,       ���E	�?L�]��A�*

A2S/average_reward_1Wf�A#���,       ���E	�vR�]��A��*

A2S/average_reward_1g#�A�0�,       ���E	����]��A�*

A2S/average_reward_1\?C
2-�,       ���E	����]��A��*

A2S/average_reward_1
Q�Ae*2,       ���E	�]��AȨ*

A2S/average_reward_1��B8�,       ���E	Ăۿ]��A��*

A2S/average_reward_1U�B�g��,       ���E	T�]�]��A��*

A2S/average_reward_1�C Q�),       ���E	�T��]��A��*

A2S/average_reward_1�CeCf��a,       ���E	�q��]��Aί*

A2S/average_reward_1hӋBfG��,       ���E	?���]��A��*

A2S/average_reward_1���C�P�5,       ���E	���]��A��*

A2S/average_reward_1i3�A����,       ���E	%���]��A��*

A2S/average_reward_1y�A�b��,       ���E	9Z��]��A��*

A2S/average_reward_16��C��h,       ���E	UT��]��A˺*

A2S/average_reward_1�	�A+~!,       ���E	�%�]��A��*

A2S/average_reward_1ͩ)D�@!,       ���E	�>�]��A��*

A2S/average_reward_1�t1B9V�,       ���E	z�J�]��A��*

A2S/average_reward_10�(B��,       ���E	)�S�]��A��*

A2S/average_reward_1��A�0j�,       ���E	D\��]��A��*

A2S/average_reward_1�Z�C]��',       ���E	*M:�]��A��*

A2S/average_reward_1��C1:,       ���E	+�I�]��A��*

A2S/average_reward_1D�B��c�,       ���E	'�|�]��A��*

A2S/average_reward_1���B����,       ���E	wz��]��A��*

A2S/average_reward_1���A�S�,       ���E	���]��A��*

A2S/average_reward_1bGkC��1,       ���E	΂�]��A��*

A2S/average_reward_1�/�C��x<,       ���E	�r�]��A��*

A2S/average_reward_1k'#BP�c,       ���E	��-�]��A��*

A2S/average_reward_1Z'6B/�,       ���E	�~~�]��A��*

A2S/average_reward_1$�)Dz�c,       ���E	�_��]��A��*

A2S/average_reward_1�\D�k9q,       ���E	`	��]��A��*

A2S/average_reward_1���C��Ю,       ���E	���]��A��*

A2S/average_reward_12I+D�-`,       ���E	}��]��A��*

A2S/average_reward_1��A���,       ���E	Ӟ-�]��A��*

A2S/average_reward_1mb�B؃݄,       ���E	y�5�]��A��*

A2S/average_reward_1h  B2�U,       ���E	��=�]��A��*

A2S/average_reward_1��A���,       ���E	
���]��A��*

A2S/average_reward_1���C��[�,       ���E	v�]��A��*

A2S/average_reward_1�L�B��^,       ���E	ɜ�]��A��*

A2S/average_reward_1ڭ�A-���,       ���E	�S/�]��A��*

A2S/average_reward_1���A�ʭ,       ���E	�L��]��A��*

A2S/average_reward_1"�mC�!�,       ���E	����]��A��*

A2S/average_reward_1g3Bz�/,       ���E	�{6�]��A��*

A2S/average_reward_1��-D�T{>,       ���E	aF�]��A��*

A2S/average_reward_1G�Bk@,       ���E	�~�]��A��*

A2S/average_reward_1���Ch���,       ���E	c,�]��A��*

A2S/average_reward_1�B�__,       ���E	b\o�]��A��*

A2S/average_reward_1�,D�{�,       ���E	-�q�]��A��*

A2S/average_reward_1>zA�?8p,       ���E	�@��]��Aѐ*

A2S/average_reward_1�?�AZ�C,       ���E	���]��A��*

A2S/average_reward_1��B\q�z,       ���E	�G��]��A��*

A2S/average_reward_1)B\9
�,       ���E	���]��Aϒ*

A2S/average_reward_11� B��:�,       ���E	���]��A��*

A2S/average_reward_1[(B�=�,       ���E	�Xk�]��A��*

A2S/average_reward_1_�)D� �,       ���E	��`�]��A��*

A2S/average_reward_1���Co�>,       ���E	�z�]��AŠ*

A2S/average_reward_1��9B�Y�,       ���E	�C��]��A�*

A2S/average_reward_1*��A�T.�,       ���E	R���]��AΨ*

A2S/average_reward_1!	,Dn�,       ���E	ֺ�]��A��*

A2S/average_reward_1&%"Bh��,       ���E	�>�]��A��*

A2S/average_reward_1��Bư�,       ���E	��]��A��*

A2S/average_reward_1^��C��,       ���E	���]��Aޯ*

A2S/average_reward_1��B�Xt,       ���E	�'.�]��A��*

A2S/average_reward_1H8BM��y,       ���E	,�2�]��A��*

A2S/average_reward_1�x�A�BÒ,       ���E	�ж�]��A��*

A2S/average_reward_1ic�C��,       ���E	����]��Aϳ*

A2S/average_reward_1�+B�d�,       ���E	O �]��A��*

A2S/average_reward_1 -DP�X,       ���E	j0�]��Aѻ*

A2S/average_reward_1:�A�g�,       ���E	�D�]��A��*

A2S/average_reward_1(:�@����,       ���E	p:�]��A��*

A2S/average_reward_1m-�B�{��,       ���E	��]��A��*

A2S/average_reward_1Q-D� I,       ���E	}$��]��A��*

A2S/average_reward_1���A�CR ,       ���E	ջ��]��A��*

A2S/average_reward_1(MB)d�s,       ���E	�m��]��A��*

A2S/average_reward_1g�AQJ2,       ���E	����]��A��*

A2S/average_reward_1)�B�0M,       ���E	��(�]��A��*

A2S/average_reward_1;.Dx��,       ���E	Z�o�]��A��*

A2S/average_reward_1��+D���,       ���E	�Z��]��A��*

A2S/average_reward_1)D�0��,       ���E	�"�]��A��*

A2S/average_reward_1tSD�WLt,       ���E	����]��A��*

A2S/average_reward_1���C���,       ���E	% �]��A��*

A2S/average_reward_1#�D�[�,       ���E	���]��A��*

A2S/average_reward_1S�@Bi�j�,       ���E	�M3�]��A��*

A2S/average_reward_1�^BO��2,       ���E	�-:�]��A��*

A2S/average_reward_1%��A�QY|,       ���E	�XI�]��A��*

A2S/average_reward_1((�AZ�;
,       ���E	�KT�]��A��*

A2S/average_reward_1Ir�A�br,       ���E	����]��A��*

A2S/average_reward_1=\C]�2S,       ���E	����]��A��*

A2S/average_reward_1�-D_��^,       ���E	&�^�]��A��*

A2S/average_reward_1��CR�3�,       ���E	�T��]��A��*

A2S/average_reward_1	0C"|�z,       ���E	D���]��A��*

A2S/average_reward_1��C�P��,       ���E	t��]��AՄ*

A2S/average_reward_1d� B�$�,       ���E	�	U�]��A��*

A2S/average_reward_1y�/DJ��,       ���E	E�`�]��A�*

A2S/average_reward_1ڏB
BG$,       ���E	�8n�]��A��*

A2S/average_reward_1Ӥ"A�97],       ���E	�{�]��A��*

A2S/average_reward_1�}B� ��,       ���E	vz��]��A�*

A2S/average_reward_1o^'B��e,       ���E	g���]��A��*

A2S/average_reward_1�F�AzJ��,       ���E	�*��]��A��*

A2S/average_reward_1R�-D.:�m,       ���E	2 ��]��A��*

A2S/average_reward_17M�A-"�,       ���E	����]��A��*

A2S/average_reward_1�A���<,       ���E	*��]��Aɖ*

A2S/average_reward_1��A�<5&,       ���E	v�]��A��*

A2S/average_reward_11SC0a��,       ���E	���]��A˙*

A2S/average_reward_1�A@�,       ���E	���]��A��*

A2S/average_reward_1?Y-D�S��,       ���E	^��]��A�*

A2S/average_reward_1��LB�W��,       ���E	�6B�]��A��*

A2S/average_reward_1AܔC����,       ���E	-�u�]��AȦ*

A2S/average_reward_1�C�SZH,       ���E	����]��A��*

A2S/average_reward_1�+C�_,�,       ���E	�k��]��A˨*

A2S/average_reward_1�@�A���e,       ���E	g��]��A��*

A2S/average_reward_1L�2C�h�X,       ���E	h�&�]��A��*

A2S/average_reward_1�g<B�,,       ���E	��,�]��A��*

A2S/average_reward_1I�A��},       ���E	G�8�]��A��*

A2S/average_reward_1h��A��,       ���E	�]��]��A��*

A2S/average_reward_1k+D� $,       ���E	C��]��A��*

A2S/average_reward_1�F'D���,       ���E	���]��A��*

A2S/average_reward_1��A{G��,       ���E	".a�]��Aľ*

A2S/average_reward_1}[�C��I�,       ���E	��k�]��A�*

A2S/average_reward_1{2*B�W��,       ���E	2lE�]��A��*

A2S/average_reward_1�$D�.��,       ���E	�`�]��A��*

A2S/average_reward_17�B x�,       ���E	�Ey�]��A��*

A2S/average_reward_1ށ<B>��,       ���E	�x��]��A��*

A2S/average_reward_1_YDSn�,       ���E	�o��]��A��*

A2S/average_reward_1<C��_�,       ���E	�4�]��A��*

A2S/average_reward_1C(-D��J�,       ���E	���]��A��*

A2S/average_reward_1�*D'��,       ���E	q���]��A��*

A2S/average_reward_1�v�AM6^�,       ���E	��]��A��*

A2S/average_reward_1�.Dz���,       ���E	YT*�]��A��*

A2S/average_reward_11�6B)q,       ���E	 :B�]��A��*

A2S/average_reward_1�J�Aс9,       ���E	��T�]��A��*

A2S/average_reward_1�10BC��,       ���E	��c�]��A��*

A2S/average_reward_1���A�7,       ���E	�Vt�]��A��*

A2S/average_reward_1BdC��,       ���E	����]��A��*

A2S/average_reward_1-5<C<b��,       ���E	�P�]��A��*

A2S/average_reward_1,3*D�&˖,       ���E	����]��A��*

A2S/average_reward_1]�,D���g,       ���E	ʗ��]��A��*

A2S/average_reward_1@��A']��,       ���E	U ��]��A� *

A2S/average_reward_1\�(DB03�,       ���E	�\��]��A�� *

A2S/average_reward_1��C�D��,       ���E	����]��A�� *

A2S/average_reward_1"�B(j�,       ���E	����]��A܇ *

A2S/average_reward_1Bo���,       ���E	���]��A܎ *

A2S/average_reward_1��DҖ�,       ���E	��l�]��AĖ *

A2S/average_reward_1k�-D�:��,       ���E	�My�]��A� *

A2S/average_reward_1$�1B^��,       ���E	i�c�]��A�� *

A2S/average_reward_1�b�Cn3�,       ���E	�9~�]��A� *

A2S/average_reward_1N�CB�?1�,       ���E	�u�]��A�� *

A2S/average_reward_1���C���,       ���E	���]��Aң *

A2S/average_reward_1X�CbJ�,       ���E	����]��A�� *

A2S/average_reward_1�V6B=�ٿ,       ���E	ȳ� ^��A�� *

A2S/average_reward_1 �C�h��,       ���E	30� ^��A�� *

A2S/average_reward_1���A�F1,       ���E	-(� ^��A�� *

A2S/average_reward_1B�CCX��L,       ���E	�� ^��AЪ *

A2S/average_reward_1�!Bʁ,       ���E	Y�S^��A�� *

A2S/average_reward_1�T(D�-BG,       ���E	��$^��A�� *

A2S/average_reward_1fe�C҅�e,       ���E	هs^��A� *

A2S/average_reward_1��)D�U�,       ���E	�K'^��A�� *

A2S/average_reward_1��Cn��,       ���E	X=3^��A�� *

A2S/average_reward_1�B2Bq٧�,       ���E	ʨ�^��A�� *

A2S/average_reward_11�+D���,       ���E	��^��A�� *

A2S/average_reward_1�Y,C�˾,       ���E	? �^��A�� *

A2S/average_reward_1 �C�qnx       ��!�	7u�^��A�� *i

A2S/kll�<

A2S/policy_network_loss�r�

A2S/value_network_lossv\C

A2S/q_network_loss|�^C'�3�,       ���E	E�^��A�� *

A2S/average_reward_1~pBW�P5,       ���E	4��^��A�� *

A2S/average_reward_1('B���',       ���E	���^��A�� *

A2S/average_reward_1�d�C�al�,       ���E	�if^��A�� *

A2S/average_reward_1v��Cnwy�,       ���E	<��^��A�� *

A2S/average_reward_1�	|C�P��,       ���E	�r2^��A�� *

A2S/average_reward_1*�FCy�u,       ���E	�W|^��A�� *

A2S/average_reward_1_�1DӸL�,       ���E	���^��A�� *

A2S/average_reward_1M4�B��I�,       ���E	��^��A�� *

A2S/average_reward_1gwkC���,       ���E	��<^��A�� *

A2S/average_reward_1�MB����,       ���E	�O�^��A�� *

A2S/average_reward_1��0D:�Xq,       ���E	���^��A�� *

A2S/average_reward_1#�4B[�QZ,       ���E	�8�^��A�� *

A2S/average_reward_1���B���_,       ���E	Z��^��A�� *

A2S/average_reward_1�	�B�:5,       ���E	h��^��A�� *

A2S/average_reward_1���A��t,       ���E	p�>^��A�� *

A2S/average_reward_1�ZC]��m,       ���E	��H^��A�� *

A2S/average_reward_1�J�A�H�,       ���E	��^��A�� *

A2S/average_reward_1YC�5O,       ���E	K�+^��A�� *

A2S/average_reward_1.5�C߈k8,       ���E	ص�^��A�� *

A2S/average_reward_1[�8C㝌J,       ���E	�\7^��Aӂ!*

A2S/average_reward_1i�C$**�,       ���E	�Q^��AΉ!*

A2S/average_reward_1	�)D��^�,       ���E	��^��A��!*

A2S/average_reward_1VQ�C�_��,       ���E	�C5^��Aˎ!*

A2S/average_reward_1[�'B?� �,       ���E	��M^��A��!*

A2S/average_reward_1eBWB�|,       ���E	��^��A�!*

A2S/average_reward_1�8D$×�,       ���E	���^��A��!*

A2S/average_reward_1[#�A��W,       ���E	� �^��Aԗ!*

A2S/average_reward_1�h�A.���,       ���E	^��A��!*

A2S/average_reward_1��AL���,       ���E	�\ ^��A��!*

A2S/average_reward_1��D���i,       ���E	2S!^��AԤ!*

A2S/average_reward_1�DU��r,       ���E	Y`}!^��Aץ!*

A2S/average_reward_1�=�B��,       ���E	\�!^��A�!*

A2S/average_reward_1���A�^��,       ���E	���!^��A��!*

A2S/average_reward_1gfC���,       ���E	Q�"^��A��!*

A2S/average_reward_1H�IB��,       ���E	��"^��A��!*

A2S/average_reward_1,�A�n�<,       ���E	2q)"^��A��!*

A2S/average_reward_1�EBG���,       ���E	�0#^��A��!*

A2S/average_reward_1JD���,       ���E	�ׅ$^��A��!*

A2S/average_reward_179D��,       ���E	���$^��A��!*

A2S/average_reward_1�x�C]eF�,       ���E	�k %^��A��!*

A2S/average_reward_1R,B�U�,       ���E	��%^��AѺ!*

A2S/average_reward_1�0�AP�l,       ���E	�@%^��A��!*

A2S/average_reward_1���A���,       ���E	Y|x&^��A��!*

A2S/average_reward_1��+Dl��,       ���E	m��&^��A��!*

A2S/average_reward_1�g�A7��,       ���E	�&^��A��!*

A2S/average_reward_1W�#B��=,       ���E	���&^��A��!*

A2S/average_reward_1�ΎA=�,       ���E	�v'^��A��!*

A2S/average_reward_1ϯ7Cʃp,       ���E	��-'^��A��!*

A2S/average_reward_1�F]B�o��,       ���E	�;'^��A��!*

A2S/average_reward_1���A^�?�,       ���E	C;J'^��A��!*

A2S/average_reward_1W B�)).,       ���E	���(^��A��!*

A2S/average_reward_1E�,D1f,       ���E	��(^��A��!*

A2S/average_reward_1uFCqGe,       ���E	t�)^��A��!*

A2S/average_reward_1b;B*ۄ^,       ���E	k�)^��A��!*

A2S/average_reward_1��D�8P�,       ���E	ٴ�)^��A��!*

A2S/average_reward_1k�FB�b�,       ���E	��)^��A��!*

A2S/average_reward_1\�VB*Y�,       ���E	?
M+^��A��!*

A2S/average_reward_1(k,D�u��,       ���E	v�,^��A��!*

A2S/average_reward_1��.D;|և,       ���E	�QQ-^��A��!*

A2S/average_reward_1�C:F��,       ���E	�a-^��A��!*

A2S/average_reward_1P�A�@�,       ���E	��.^��A��!*

A2S/average_reward_1%�D%�,       ���E	jW�.^��A��!*

A2S/average_reward_1%$|Ay��,       ���E	Z��.^��A��!*

A2S/average_reward_1 "B|���,       ���E	2��.^��A��!*

A2S/average_reward_12	A��Tm,       ���E	k�.^��A��!*

A2S/average_reward_1�kB]Ѣ^,       ���E	2�R/^��A��!*

A2S/average_reward_1v8�C��Iv,       ���E	d�0^��A��!*

A2S/average_reward_1ڀ(D�0��,       ���E	d��0^��A��!*

A2S/average_reward_1�U�A�Gu�,       ���E	�?2^��Aӆ"*

A2S/average_reward_1PZ,D+^Y,       ���E	y|2^��Aŉ"*

A2S/average_reward_1�G�C�ʚj,       ���E	T��2^��Aى"*

A2S/average_reward_1�@<A���Z,       ���E	��2^��A��"*

A2S/average_reward_1c�`B�3ig,       ���E	�ܤ2^��A̊"*

A2S/average_reward_1�nB,-,       ���E	;��3^��A��"*

A2S/average_reward_1�:D���,       ���E	_�M5^��A��"*

A2S/average_reward_1��.D�c�,       ���E	��g5^��Aڙ"*

A2S/average_reward_1�8�AH��,       ���E	ގr5^��A��"*

A2S/average_reward_1��A�� 3,       ���E	H/�5^��A��"*

A2S/average_reward_1�*[A3���,       ���E	޶5^��A��"*

A2S/average_reward_1��B���,       ���E	���5^��Aޛ"*

A2S/average_reward_14�EBh#�,       ���E	\��5^��A��"*

A2S/average_reward_1J�B;�d�,       ���E	�a7^��A��"*

A2S/average_reward_1H�*D<M��,       ���E	��z7^��Aʤ"*

A2S/average_reward_1� NB�V�!,       ���E	�|�8^��A��"*

A2S/average_reward_1n�)D�KW,       ���E	P�:^��A��"*

A2S/average_reward_1}�+D��e;,       ���E	QG:^��A��"*

A2S/average_reward_1���A�!�,       ���E	4�U;^��A��"*

A2S/average_reward_1.�-DC�z,       ���E	U��<^��A��"*

A2S/average_reward_1+~#DÅm<,       ���E	�8�<^��A��"*

A2S/average_reward_1UK)B5_�,       ���E	�D=^��A��"*

A2S/average_reward_1�ߟC��v�,       ���E	]�=^��A��"*

A2S/average_reward_1K�%C�F=,       ���E	�
�=^��A��"*

A2S/average_reward_1vg�Aa��{,       ���E	#�=^��A��"*

A2S/average_reward_1�M�B&x�{,       ���E	�Z�>^��A��"*

A2S/average_reward_15��C�{A,       ���E	{Y�>^��A��"*

A2S/average_reward_1#�DCP>��,       ���E	��'@^��A��"*

A2S/average_reward_1&�+D�J��,       ���E	�{�A^��A��"*

A2S/average_reward_1��*D����,       ���E	S��B^��A��"*

A2S/average_reward_1��+DT��L,       ���E	�0C^��A��"*

A2S/average_reward_1�C��l,       ���E	o�C^��A��"*

A2S/average_reward_1B�*C+��D,       ���E	�]E^��A��"*

A2S/average_reward_1ѹ+D���,       ���E	C�{F^��A��"*

A2S/average_reward_1\�+D�K

,       ���E	+�F^��A��"*

A2S/average_reward_1�R�@: ��,       ���E	���F^��A��"*

A2S/average_reward_1WI@C���,       ���E	��G^��A��"*

A2S/average_reward_1U�A��T,       ���E	�G^��A��"*

A2S/average_reward_1j��AW�[,       ���E	LAG^��A��"*

A2S/average_reward_1�q�AW�s�,       ���E	��G^��A��#*

A2S/average_reward_1x�AG�:!,       ���E	�\�G^��AŃ#*

A2S/average_reward_1�l�C���,       ���E	��H^��AՅ#*

A2S/average_reward_1�CCvY5 ,       ���E	>�H^��A��#*

A2S/average_reward_1�NUB�v	8,       ���E	�v�I^��A��#*

A2S/average_reward_1�-Dg�k�,       ���E	�)�I^��AƎ#*

A2S/average_reward_1֨�A �D,       ���E	*K^��A��#*

A2S/average_reward_1ɥ.D")�,       ���E	U �K^��A��#*

A2S/average_reward_1*�=C�@(,       ���E	>��K^��A��#*

A2S/average_reward_1a�,ChdY�,       ���E	�?�K^��A�#*

A2S/average_reward_1�B���,       ���E	�̵L^��A��#*

A2S/average_reward_1��C�'?,       ���E	S��L^��A#*

A2S/average_reward_18<�A��g,       ���E	��M^��A�#*

A2S/average_reward_1$��C%�,       ���E	�R�M^��Aǥ#*

A2S/average_reward_1r�BNe0W,       ���E	���M^��A�#*

A2S/average_reward_1���A���-x       ��!�	��
Y^��A�#*i

A2S/kl�T9

A2S/policy_network_loss�O��

A2S/value_network_loss*VC

A2S/q_network_loss�WC���,       ���E	��`Z^��A˭#*

A2S/average_reward_1�6D�] �,       ���E	�jZ^��A��#*

A2S/average_reward_1�B�o`�,       ���E	�&rZ^��A��#*

A2S/average_reward_1�K�A���,       ���E	��I[^��A��#*

A2S/average_reward_1���C�74,       ���E	DAQ[^��AӲ#*

A2S/average_reward_1$��AwWP�,       ���E	�T�[^��A��#*

A2S/average_reward_1���BM���,       ���E	���\^��A�#*

A2S/average_reward_1d+D�@�,       ���E	+^^��A��#*

A2S/average_reward_1�f2D:��,       ���E	Ϭ$^^��A��#*

A2S/average_reward_1�cB�)�k,       ���E	�-_^��A��#*

A2S/average_reward_1X�D��Wk,       ���E	$��_^��A��#*

A2S/average_reward_1���CЭ�`,       ���E	7��_^��A��#*

A2S/average_reward_1��9B�[O�,       ���E	��_^��A��#*

A2S/average_reward_1�'B��wH,       ���E	�F�_^��A��#*

A2S/average_reward_1�%6B^��,       ���E	��N`^��A��#*

A2S/average_reward_1A�kCjy!,       ���E	W׭`^��A��#*

A2S/average_reward_1�}LC\�,       ���E	���`^��A��#*

A2S/average_reward_1n�"Bƈ��,       ���E	�b^��A��#*

A2S/average_reward_1��6D]��,       ���E	^�b^��A��#*

A2S/average_reward_1]�3BQ�,       ���E	C	�b^��A��#*

A2S/average_reward_1�וC @�,       ���E	;��b^��A��#*

A2S/average_reward_1��NB���,       ���E	lf�b^��A��#*

A2S/average_reward_1��A���,       ���E	!��c^��A��#*

A2S/average_reward_1	��C��~�,       ���E	��d^��A��#*

A2S/average_reward_1�3Dr}J,       ���E	n��e^��A��#*

A2S/average_reward_1#��CRj(,       ���E	2�1g^��A��#*

A2S/average_reward_1�t2DE�4,       ���E	���h^��A��#*

A2S/average_reward_18�.D�}&O,       ���E	��h^��A��$*

A2S/average_reward_1e�B!���,       ���E	e��h^��A��$*

A2S/average_reward_1S��B��em,       ���E	��j^��A��$*

A2S/average_reward_1�80D��˲,       ���E	�@�j^��A��$*

A2S/average_reward_1Pn�B&�y,       ���E	��l^��A��$*

A2S/average_reward_1e�0D��b�,       ���E	��#l^��A��$*

A2S/average_reward_1i��A�W�,       ���E	z6l^��A�$*

A2S/average_reward_16�TA"��;,       ���E	�3@l^��A��$*

A2S/average_reward_1{*�A����,       ���E	�8tm^��A�$*

A2S/average_reward_1��.D�)�,       ���E	/ԅm^��A��$*

A2S/average_reward_1�r�RwRl,       ���E	2"�m^��Aʛ$*

A2S/average_reward_133<@�� #,       ���E	z�m^��A؛$*

A2S/average_reward_1ᨢA˖R,       ���E	A��m^��A��$*

A2S/average_reward_1.wzB�rYx,       ���E	]]An^��AǞ$*

A2S/average_reward_1��cC�`7�,       ���E	��Un^��A��$*

A2S/average_reward_1A�Bm1�x       ��!�	 Jy^��A��$*i

A2S/kl�?�8

A2S/policy_network_loss�广

A2S/value_network_loss�xC

A2S/q_network_loss��wC��N),       ���E	i1Xy^��A��$*

A2S/average_reward_1�V�A��,       ���E	 bjy^��Aޟ$*

A2S/average_reward_1�2lB�g�,       ���E	�ֻy^��Aѡ$*

A2S/average_reward_1U*CB5j,       ���E	1�&z^��A��$*

A2S/average_reward_1아C�1��,       ���E	��mz^��A��$*

A2S/average_reward_1'#Cl�X@,       ���E		Ԡ{^��A��$*

A2S/average_reward_1��D���,       ���E	I��{^��AЬ$*

A2S/average_reward_1DT�B��*q,       ���E	�\�{^��A��$*

A2S/average_reward_1��B1��,       ���E	x|^��Aʮ$*

A2S/average_reward_1�,�B�OD,       ���E	���|^��A��$*

A2S/average_reward_1jγC��+,       ���E	�-�|^��Aڲ$*

A2S/average_reward_1
DBڨ�,       ���E	���|^��A�$*

A2S/average_reward_15SA��,       ���E	��|^��A��$*

A2S/average_reward_1P��A��Q�,       ���E	�R�|^��A��$*

A2S/average_reward_1��A�<L,       ���E	)v}^��A��$*

A2S/average_reward_1�UB�q�,       ���E	7}^��A��$*

A2S/average_reward_1���A��-b,       ���E	ψ<}^��A��$*

A2S/average_reward_1h��B����,       ���E	�H}^��A��$*

A2S/average_reward_1�B��y,       ���E	�X�}^��A��$*

A2S/average_reward_13�C���,       ���E	Ϧ}^��A��$*

A2S/average_reward_1�GB�Iyk,       ���E	��~^��A��$*

A2S/average_reward_1��IC�y,       ���E	�b\^��A��$*

A2S/average_reward_1�4DɦM�,       ���E	5�h^��A��$*

A2S/average_reward_11@BA���,       ���E	��p^��A��$*

A2S/average_reward_1��A�,�4,       ���E	��}^��A��$*

A2S/average_reward_1z	�A땑x,       ���E	��ƀ^��A��$*

A2S/average_reward_1^9D�ދ�,       ���E	��k�^��A��$*

A2S/average_reward_1���C���,       ���E	\��^��A��$*

A2S/average_reward_1HU�B�),       ���E	U��^��A��$*

A2S/average_reward_1�BO�[,       ���E	pv�^��A��$*

A2S/average_reward_1-&C���,       ���E	�㬂^��A��$*

A2S/average_reward_1��C%���,       ���E	{Q%�^��A��$*

A2S/average_reward_1�F{C��,       ���E	�1�^��A��$*

A2S/average_reward_1�DA�5uD,       ���E	�O]�^��A��$*

A2S/average_reward_1��B���D,       ���E	±g�^��A��$*

A2S/average_reward_1N+@��,       ���E	2�փ^��A��$*

A2S/average_reward_1�C���,       ���E	�p��^��A��$*

A2S/average_reward_19��B�x(�,       ���E	�|7�^��A��$*

A2S/average_reward_1d9 C��,       ���E	�<]�^��A��$*

A2S/average_reward_1c��B�fx,       ���E	�i�^��A��$*

A2S/average_reward_1�$B2.�,       ���E	��q�^��A��$*

A2S/average_reward_1�8VAz�Y,       ���E	G }�^��A��$*

A2S/average_reward_1\��A�v��,       ���E	 (��^��A��$*

A2S/average_reward_1���AZ�?=,       ���E	����^��A��$*

A2S/average_reward_1��'B~t,       ���E	���^��A��$*

A2S/average_reward_1w־A��,       ���E	�i/�^��A��$*

A2S/average_reward_1��,DiH,       ���E	;͇�^��A��$*

A2S/average_reward_1?](D(n.O,       ���E	��^��A��$*

A2S/average_reward_1�c�C+��,       ���E	c��^��A��$*

A2S/average_reward_1w�A1#�A,       ���E	/��^��A��$*

A2S/average_reward_1�:C��`�,       ���E	B���^��A��$*

A2S/average_reward_11�HB촌�,       ���E	[&a�^��A��%*

A2S/average_reward_1~|,D��O�,       ���E	i�j�^��A��%*

A2S/average_reward_1�S�A��-�,       ���E	��ފ^��Aۄ%*

A2S/average_reward_1�/oC�S@�,       ���E	���^��A��%*

A2S/average_reward_1��A��6�,       ���E	��^��A�%*

A2S/average_reward_1�EsB	I�[,       ���E	֫�^��A�%*

A2S/average_reward_1�V�C�w�>,       ���E	_�W�^��A̌%*

A2S/average_reward_1��Cu�f�,       ���E	��k�^��A��%*

A2S/average_reward_1�kB�%ɷ,       ���E	C�ӌ^��A�%*

A2S/average_reward_1޿xC��R,       ���E	���^��A֗%*

A2S/average_reward_1-�-D_P
�,       ���E	����^��A�%*

A2S/average_reward_1΀@C>��#,       ���E	w��^��Aڡ%*

A2S/average_reward_1�>*D���,       ���E	��!�^��A��%*

A2S/average_reward_1ݳC{�|�,       ���E	}�:�^��Aۣ%*

A2S/average_reward_1�6eBk!�,       ���E	�vT�^��A�%*

A2S/average_reward_1ID_Q�,       ���E	4�]�^��A��%*

A2S/average_reward_1�R�A��f�,       ���E	r���^��A�%*

A2S/average_reward_1�+D�tm,       ���E	��Ē^��A��%*

A2S/average_reward_1L��A*�F�,       ���E	���^��A��%*

A2S/average_reward_1Q�+D_%F,       ���E	��{�^��A��%*

A2S/average_reward_1��*D����,       ���E	�O��^��A��%*

A2S/average_reward_1\~#B�f,       ���E	J�ŕ^��A��%*

A2S/average_reward_1�m�BNP�,       ���E	�)͕^��A��%*

A2S/average_reward_1���A�?B,       ���E	k�Օ^��A��%*

A2S/average_reward_1�ƫA�|�,       ���E	����^��A��%*

A2S/average_reward_1��B�ݤ,       ���E	�o��^��A��%*

A2S/average_reward_1P��C� ��,       ���E	����^��A��%*

A2S/average_reward_1��A���,       ���E	���^��A��%*

A2S/average_reward_1Ľ�An�1c,       ���E	w%��^��A��%*

A2S/average_reward_1f�-DXW�,       ���E	tķ�^��A��%*

A2S/average_reward_1�v�A��[,       ���E	6%�^��A��%*

A2S/average_reward_1J�&D��2,       ���E	*u6�^��A��%*

A2S/average_reward_1�B�ݔ,       ���E	6N��^��A��%*

A2S/average_reward_1��)CDkbW,       ���E	���^��A��%*

A2S/average_reward_1~�A����x       ��!�	E�\�^��A��%*i

A2S/klU��7

A2S/policy_network_loss�@��

A2S/value_network_lossK�\C

A2S/q_network_loss.aC+�`,       ���E	pk�^��A��%*

A2S/average_reward_1�ZAB��,       ���E	����^��A��%*

A2S/average_reward_1�zB�S�o,       ���E	9��^��A��%*

A2S/average_reward_1�/9D����,       ���E	��H�^��A��%*

A2S/average_reward_1�z3Ct8ֆ,       ���E	`\�^��A��%*

A2S/average_reward_1HFBﴭ`,       ���E	���^��A��%*

A2S/average_reward_1��C���m,       ���E	�w�^��A��%*

A2S/average_reward_1�ԚC�w��,       ���E	)ي�^��A��%*

A2S/average_reward_1
ϷAQ�^,       ���E	\8�^��A��%*

A2S/average_reward_1_��C��,       ���E	κ��^��A��%*

A2S/average_reward_1h�kC�*�,       ���E	�'��^��A��%*

A2S/average_reward_1�t	B��u,       ���E	�ͩ^��A��%*

A2S/average_reward_1n�Bڪ�G,       ���E	pݩ^��A��%*

A2S/average_reward_1`MB��8�,       ���E	վj�^��A��%*

A2S/average_reward_1Ж�C�jڕ,       ���E	�s�^��A��%*

A2S/average_reward_1���A���k,       ���E	����^��A��%*

A2S/average_reward_1��A)$�,       ���E	���^��A��%*

A2S/average_reward_1�'BK>�,       ���E	��ɪ^��A��%*

A2S/average_reward_1
��B�J,       ���E		Ԫ^��A��%*

A2S/average_reward_1� BGsM�,       ���E	|ݪ^��A��%*

A2S/average_reward_1}��A^v�V,       ���E	c�T�^��AՀ&*

A2S/average_reward_1�T�Ci��r,       ���E	��Ԭ^��A��&*

A2S/average_reward_1r`-D�U,       ���E	2r�^��A��&*

A2S/average_reward_1�CC����,       ���E	G*�^��A��&*

A2S/average_reward_1= &B�gG,       ���E	�Ls�^��A��&*

A2S/average_reward_1�d+C=b�,       ���E	�I��^��A��&*

A2S/average_reward_1�3�A	|�,       ���E	�U�^��A��&*

A2S/average_reward_1��2Dx���,       ���E	�U��^��A��&*

A2S/average_reward_1���C�}q,       ���E	�P��^��Aݘ&*

A2S/average_reward_1~��?�×=,       ���E	J��^��A��&*

A2S/average_reward_1��C��Ğ,       ���E	��^��A��&*

A2S/average_reward_1���A��v�,       ���E	O���^��A�&*

A2S/average_reward_1I�fAi���,       ���E	���^��A��&*

A2S/average_reward_1��?B+Ƣ),       ���E	�b�^��A��&*

A2S/average_reward_1��&D�G��,       ���E	��y�^��AѢ&*

A2S/average_reward_16�A��{�,       ���E	t臱^��A�&*

A2S/average_reward_1�BNӵ�,       ���E	����^��A��&*

A2S/average_reward_1���A�>��,       ���E	����^��Aޣ&*

A2S/average_reward_1tfJB��C,       ���E	��^��Aפ&*

A2S/average_reward_1{8�BVFpO,       ���E	���^��A֥&*

A2S/average_reward_1b��B�,       ���E	8���^��A��&*

A2S/average_reward_1	�gCֺ%�,       ���E	����^��A��&*

A2S/average_reward_1��B��b,       ���E	d��^��A��&*

A2S/average_reward_1� �A�ۃ�,       ���E	�LĲ^��Aө&*

A2S/average_reward_1rhB��@,       ���E	�EԲ^��A��&*

A2S/average_reward_1�J&B�6�,       ���E	ܰݲ^��A��&*

A2S/average_reward_1ٴ�A'��,       ���E	N��^��A��&*

A2S/average_reward_1��A$�,       ���E	��6�^��A��&*

A2S/average_reward_1<+D�~k�,       ���E	T\�^��A��&*

A2S/average_reward_1)sB
m�,       ���E	�8m�^��A��&*

A2S/average_reward_1Ӣ'BZ�ƅ,       ���E	*���^��A�&*

A2S/average_reward_1���Aޏ",       ���E	MG��^��AŴ&*

A2S/average_reward_1X3fB��w�,       ���E	�t��^��A��&*

A2S/average_reward_1U?/BY��,       ���E	<fӵ^��A޻&*

A2S/average_reward_1�5DYٱ},       ���E	E-�^��A׽&*

A2S/average_reward_1�&Cj�W,       ���E	��l�^��A��&*

A2S/average_reward_1^C�0�,       ���E	��w�^��AͿ&*

A2S/average_reward_1�Si@�!�g,       ���E	i^��^��A��&*

A2S/average_reward_1-D�A�,       ���E	~η^��A��&*

A2S/average_reward_1�B��(,       ���E	�n�^��A��&*

A2S/average_reward_1u�eB�&��,       ���E	.���^��A��&*

A2S/average_reward_1GqB�nZ�,       ���E	*�׸^��A��&*

A2S/average_reward_1[@�C��,       ���E	�+�^��A��&*

A2S/average_reward_1R�*D�3,       ���E	(�4�^��A��&*

A2S/average_reward_1%u�A�}ƈ,       ���E	�VS�^��A��&*

A2S/average_reward_1�׃B�|,       ���E		v]�^��A��&*

A2S/average_reward_1)��A�E��,       ���E	�>c�^��A��&*

A2S/average_reward_1'��A�n��,       ���E	�1m�^��A��&*

A2S/average_reward_1}�A�^�,       ���E	�q}�^��A��&*

A2S/average_reward_1z�A���,       ���E	w�]�^��A��&*

A2S/average_reward_1���Ca���,       ���E	1g�^��A��&*

A2S/average_reward_1g̈AE��,       ���E	
���^��A��&*

A2S/average_reward_1��$C�k��,       ���E	�$�^��A��&*

A2S/average_reward_1�$dC�WCX,       ���E	w,w�^��A��&*

A2S/average_reward_1�2.Dr�jp,       ���E	w ��^��A��&*

A2S/average_reward_1�O*B�c��,       ���E	�~��^��A��&*

A2S/average_reward_1��B9�T,       ���E	H���^��A��&*

A2S/average_reward_1�mD����,       ���E	!m��^��A��&*

A2S/average_reward_1(�/D�ց?,       ���E	hS˿^��A��&*

A2S/average_reward_1���A3!#�,       ���E	\f>�^��A��&*

A2S/average_reward_1a�LC΀�6,       ���E	�%R�^��A��&*

A2S/average_reward_1�H0B���,       ���E	7�L�^��A��&*

A2S/average_reward_1�*�C�e��,       ���E	ՌM�^��A̅'*

A2S/average_reward_1��D��א,       ���E	ηV�^��A�'*

A2S/average_reward_1 �A��K,       ���E	��k�^��A��'*

A2S/average_reward_1�B��,       ���E	,���^��A�'*

A2S/average_reward_1��"D_Ή�,       ���E	L�n�^��A��'*

A2S/average_reward_19�C���,       ���E	=��^��Aߒ'*

A2S/average_reward_1�$C�j�Q,       ���E	����^��A��'*

A2S/average_reward_1J�NB�p��,       ���E	�;��^��Aۓ'*

A2S/average_reward_1��PB���,       ���E	����^��A��'*

A2S/average_reward_1`��C�O��,       ���E	lL=�^��A��'*

A2S/average_reward_1ELC���V,       ���E	{�H�^��Aۛ'*

A2S/average_reward_1|FA�:��,       ���E	�Lb�^��A��'*

A2S/average_reward_1F�PB,��,       ���E	a0��^��A�'*

A2S/average_reward_1Wf B��l,       ���E	�^��^��A͢'*

A2S/average_reward_1�E�CE�P,       ���E	�8r�^��A̧'*

A2S/average_reward_1��C@S��,       ���E	����^��A��'*

A2S/average_reward_1��B]o��,       ���E	��^��A�'*

A2S/average_reward_1�*D��.�,       ���E	cf(�^��A��'*

A2S/average_reward_1��A�U�v,       ���E	�Q�^��A��'*

A2S/average_reward_1=��Bg���,       ���E	����^��A��'*

A2S/average_reward_1��BC�#:*,       ���E	B�^��A��'*

A2S/average_reward_1_�-D#�=�,       ���E	�3�^��A��'*

A2S/average_reward_1��A���
,       ���E	�<�^��A��'*

A2S/average_reward_1=��AV�C,       ���E	h �^��AƼ'*

A2S/average_reward_1��5A�Bb�,       ���E	 �+�^��A�'*

A2S/average_reward_1�?B�
�m,       ���E	k�3�^��A��'*

A2S/average_reward_1!.�Aq�f,       ���E	=ε�^��A��'*

A2S/average_reward_1]C��),       ���E	ڵ�^��A��'*

A2S/average_reward_1�Q%D���*,       ���E	h6x�^��A��'*

A2S/average_reward_1��,D�@ ,       ���E	����^��A��'*

A2S/average_reward_1)^C=h�%,       ���E	���^��A��'*

A2S/average_reward_1p{DO-�,       ���E	��.�^��A��'*

A2S/average_reward_1>e-B�9,       ���E	i8�^��A��'*

A2S/average_reward_1P��A[���,       ���E	�{F�^��A��'*

A2S/average_reward_15[B�I�,       ���E	���^��A��'*

A2S/average_reward_1|)D}>i,       ���E	Z<��^��A��'*

A2S/average_reward_1��A"#�,       ���E	�z��^��A��'*

A2S/average_reward_1���A~4fc,       ���E	�T�^��A��'*

A2S/average_reward_1cb,D���,       ���E	��/�^��A��'*

A2S/average_reward_1�BI�S},       ���E	/>�^��A��'*

A2S/average_reward_1*�B�T�K,       ���E	�n��^��A��'*

A2S/average_reward_1��GC��f�,       ���E	�$ �^��A��'*

A2S/average_reward_1�,D e!�,       ���E	����^��A��'*

A2S/average_reward_1�E�C�z�,       ���E	T��^��A��'*

A2S/average_reward_1�^B�f�,       ���E	0��^��A��'*

A2S/average_reward_1=��C�H�,       ���E	׎�^��A��'*

A2S/average_reward_1�:�C&'<,       ���E	��6�^��A��(*

A2S/average_reward_1��MB��ϫ,       ���E	���^��A��(*

A2S/average_reward_1Q��C"HĦ,       ���E	��^��A�(*

A2S/average_reward_1#�A�;+6,       ���E	O!�^��A��(*

A2S/average_reward_1�1�A3~]�,       ���E	g9�^��AΆ(*

A2S/average_reward_1�l"B��2,       ���E	�B�^��A�(*

A2S/average_reward_1ƹ�A�,,       ���E	���^��A��(*

A2S/average_reward_1ǭ�Cט��,       ���E	�E�^��A؊(*

A2S/average_reward_1�͋B�s�,       ���E	�Yg�^��A��(*

A2S/average_reward_1�7C�c��,       ���E	!)�^��Aʏ(*

A2S/average_reward_1�I�C���=,       ���E	�
H�^��A��(*

A2S/average_reward_1�,D���,       ���E	^��^��A��(*

A2S/average_reward_1q�C��u�,       ���E	����^��Aܙ(*

A2S/average_reward_1=�BBM!I,       ���E	���^��A��(*

A2S/average_reward_1��cA#AN�,       ���E	�*��^��A��(*

A2S/average_reward_1��A����,       ���E	 `��^��A�(*

A2S/average_reward_1�C}�fg,       ���E	��^��A��(*

A2S/average_reward_1Τ�A�@��,       ���E	�~��^��A��(*

A2S/average_reward_1'w�B�][,       ���E	�_��^��A��(*

A2S/average_reward_1���A����,       ���E	���^��A�(*

A2S/average_reward_1��nB���,       ���E	�a*�^��A��(*

A2S/average_reward_1��A��,       ���E	���^��AӤ(*

A2S/average_reward_1��C�>�,       ���E	_���^��A��(*

A2S/average_reward_1�<�A�k��,       ���E	����^��A��(*

A2S/average_reward_1|�AB?}߈,       ���E	Es'�^��A��(*

A2S/average_reward_1/M,D��1�,       ���E	"�7�^��Aĭ(*

A2S/average_reward_1J(�A�J~h,       ���E	*�D�^��A�(*

A2S/average_reward_1\��A�IO*,       ���E	o�P�^��A��(*

A2S/average_reward_1���A ���,       ���E		z��^��A�(*

A2S/average_reward_1�5+D(/�,       ���E	EP�^��A��(*

A2S/average_reward_1� �C<3�,       ���E	��,�^��A��(*

A2S/average_reward_15b�C}�I,       ���E	Σ6�^��A��(*

A2S/average_reward_1�W�AX��s,       ���E	:d�^��A��(*

A2S/average_reward_1�D �� ,       ���E	"G��^��A��(*

A2S/average_reward_1�n�C2Df�,       ���E	u��^��A��(*

A2S/average_reward_1�f�CE�}j,       ���E	<>�^��A��(*

A2S/average_reward_1	AB3V��,       ���E	�y�^��A��(*

A2S/average_reward_1\��A+%�R,       ���E	ܗ�^��A��(*

A2S/average_reward_1�B�RXr,       ���E	���^��A��(*

A2S/average_reward_1NܝC��
,       ���E	����^��A��(*

A2S/average_reward_1T��A�(Y,       ���E		���^��A��(*

A2S/average_reward_1�S�C~� �,       ���E	����^��A��(*

A2S/average_reward_1�҇BI�ݽ,       ���E	j��^��A��(*

A2S/average_reward_1��A�T�,       ���E	����^��A��(*

A2S/average_reward_1��AA؂�,       ���E	�^��^��A��(*

A2S/average_reward_19ڝB����,       ���E	^*�^��A��(*

A2S/average_reward_1�JD#؇<,       ���E	�^��A��(*

A2S/average_reward_1��-D�߽,       ���E	����^��A��(*

A2S/average_reward_1�ܚAg��,       ���E	$�"�^��A��(*

A2S/average_reward_1�WKCDO��,       ���E	�&.�^��A��(*

A2S/average_reward_176�@�AX_,       ���E	����^��A��(*

A2S/average_reward_1�C��6,       ���E	����^��A��(*

A2S/average_reward_1���Au@��,       ���E	FU��^��A��(*

A2S/average_reward_1'�A=݉,       ���E	(c?�^��A��(*

A2S/average_reward_1��)C�k�o,       ���E	�N��^��A��(*

A2S/average_reward_1�rfC�5��,       ���E	@��^��A��(*

A2S/average_reward_1ޱC��L,       ���E	��^��A��(*

A2S/average_reward_1�'Df���,       ���E	��^�^��A��)*

A2S/average_reward_12O.Dq���,       ���E	9�o�^��A�)*

A2S/average_reward_1b�LB (�,       ���E	ARz�^��A��)*

A2S/average_reward_1��@b`�,       ���E	�A'�^��A��)*

A2S/average_reward_1ꞧC���(,       ���E	�3?�^��A�)*

A2S/average_reward_1�>WB0�z;,       ���E	,#W�^��A��)*

A2S/average_reward_1Ҭ~B�-3�,       ���E	@�b�^��Aڊ)*

A2S/average_reward_1�f�ABG�d,       ���E	piu�^��A��)*

A2S/average_reward_1mB3�;.,       ���E	�l��^��Aϑ)*

A2S/average_reward_1��
D��D>,       ���E	!���^��A��)*

A2S/average_reward_1�CҾF�,       ���E	��^��A��)*

A2S/average_reward_1��A�1,       ���E	�=�^��A��)*

A2S/average_reward_1�b+DK�x�,       ���E	�jU�^��Aݛ)*

A2S/average_reward_1��~BW�+,       ���E	x�d�^��A��)*

A2S/average_reward_1ܨB�t�:,       ���E	�ed�^��Aȡ)*

A2S/average_reward_1W��C��,       ���E	��}�^��A��)*

A2S/average_reward_1��]BB7�,       ���E	�&��^��A��)*

A2S/average_reward_19Q�A�e��,       ���E	o!�^��A��)*

A2S/average_reward_1
��CE�Cq,       ���E	D�4�^��A�)*

A2S/average_reward_1d�"B2-��,       ���E	U�^�^��A��)*

A2S/average_reward_1L�
D ��Y,       ���E	�Ks�^��Aɬ)*

A2S/average_reward_1��-B���Q,       ���E	Tb��^��A��)*

A2S/average_reward_1�JB
-F,       ���E	aǔ�^��A��)*

A2S/average_reward_1��Ar���,       ���E	�~��^��A��)*

A2S/average_reward_1��Af��M,       ���E	 ��^��AӮ)*

A2S/average_reward_1�ۅA'U�,       ���E	:��^��A�)*

A2S/average_reward_1���A��|z,       ���E	;D�^��Aֶ)*

A2S/average_reward_1�Q/D��,       ���E	��M�^��A�)*

A2S/average_reward_1���A6
Qx,       ���E	'bW�^��A��)*

A2S/average_reward_18K�A�#,       ���E	A�[�^��A��)*

A2S/average_reward_1���A��,       ���E	��f�^��A��)*

A2S/average_reward_1���A%�#�,       ���E	U��^��A��)*

A2S/average_reward_1{��BQ/ �,       ���E	6)�^��A��)*

A2S/average_reward_1�.�CD�Z�,       ���E	�B��^��A��)*

A2S/average_reward_1̯�C��,       ���E	n ��^��A��)*

A2S/average_reward_1�9B�j�,       ���E	�R��^��A��)*

A2S/average_reward_1)i�CeZ�,       ���E	��^��A��)*

A2S/average_reward_1&K�B`�,       ���E	�3D�^��A��)*

A2S/average_reward_1�,D����,       ���E	��b�^��A��)*

A2S/average_reward_1*BwBq�i�,       ���E	8p�^��A��)*

A2S/average_reward_1���A��x       ��!�	{��_��A��)*i

A2S/kl~�d<

A2S/policy_network_lossvN7�

A2S/value_network_loss+\C

A2S/q_network_loss��aC��9�,       ���E	�r_��A��)*

A2S/average_reward_1AB�~J\,       ���E	M<_��A��)*

A2S/average_reward_1�x�A�z,       ���E	|o_��A��)*

A2S/average_reward_1�SOC�ݮ�,       ���E	>|_��A��)*

A2S/average_reward_1��)B8�(],       ���E	�H�_��A��)*

A2S/average_reward_1� B��k�,       ���E	�r�_��A��)*

A2S/average_reward_1AM�A��� ,       ���E	��_��A��)*

A2S/average_reward_1�$�A5�^,       ���E	�j�_��A��)*

A2S/average_reward_1DX+B��,       ���E	�;�_��A��)*

A2S/average_reward_1��cC	��1,       ���E	/��_��A��)*

A2S/average_reward_1Q��AQ��,       ���E	�u_��A��)*

A2S/average_reward_1�A��
O,       ���E	�%_��A��)*

A2S/average_reward_1�&�A���\,       ���E	R(_��A��)*

A2S/average_reward_1w�;B��,       ���E	��1_��A��)*

A2S/average_reward_1�k�A)S��,       ���E	�5N_��A��)*

A2S/average_reward_1P��BKT�a,       ���E	UU_��A��)*

A2S/average_reward_1ݩ�A���,       ���E	�x�_��A��)*

A2S/average_reward_1h�0C,E,       ���E	F�	_��A��)*

A2S/average_reward_1�.{C�],       ���E	T3:	_��A��)*

A2S/average_reward_1<p�B\�b�,       ���E	b�?	_��A��)*

A2S/average_reward_11v�A|�,       ���E	�N	_��A��)*

A2S/average_reward_1���Ag��
,       ���E	f<_	_��A��)*

A2S/average_reward_1�YB�%�,       ���E	}e	_��A��)*

A2S/average_reward_1FntA��%[,       ���E	o�n	_��A��)*

A2S/average_reward_1N�B�s�h,       ���E	��	_��A��)*

A2S/average_reward_1��B'3�D,       ���E	m��	_��A��)*

A2S/average_reward_1
G.C�,|n,       ���E	�U�	_��A��)*

A2S/average_reward_1:oBac#,       ���E	�t
_��A��)*

A2S/average_reward_1vٜB�/K�,       ���E	��7
_��A��)*

A2S/average_reward_1a)�B"~E,       ���E	#3>
_��A��)*

A2S/average_reward_1D��A�>,       ���E	��F
_��A��)*

A2S/average_reward_1���A�2,       ���E	ȩW
_��A��)*

A2S/average_reward_1�>�A�v�~,       ���E	rZi
_��A��)*

A2S/average_reward_1��%B�ڪ%,       ���E	O��
_��A��)*

A2S/average_reward_1O�hBWu?,       ���E	_��
_��A��)*

A2S/average_reward_1�%�B;�,       ���E	e�
_��A��)*

A2S/average_reward_1pŬBA�,       ���E	*�
_��A��)*

A2S/average_reward_1	�B�I�,       ���E	��
_��A��)*

A2S/average_reward_1ݏBG�&�,       ���E	��_��A��)*

A2S/average_reward_1!��B�R��,       ���E	��_��A��)*

A2S/average_reward_1�(�Ae��o,       ���E	5"_��A��)*

A2S/average_reward_1p��A#]�],       ���E	�DA_��A��)*

A2S/average_reward_1�B���,       ���E	�R_��A��)*

A2S/average_reward_1�%IB�+6�,       ���E	V^_��A��)*

A2S/average_reward_1dӻA��f,       ���E	�/�_��A��)*

A2S/average_reward_1�	nC*ǠI,       ���E	<m�_��A��)*

A2S/average_reward_1VzAS�u�,       ���E	HG�_��A��)*

A2S/average_reward_1@C�B&z�,       ���E	���_��A��)*

A2S/average_reward_1�]�A]?5K,       ���E	\�7_��A��)*

A2S/average_reward_1��-D�H�h,       ���E	��_��A��)*

A2S/average_reward_1w,D�^�,       ���E	��_��A��)*

A2S/average_reward_1�Y�B�.0,       ���E	��_��A�**

A2S/average_reward_1'�+D��Q,       ���E	�"+_��A��**

A2S/average_reward_1Oz�B� O�,       ���E	=1_��A҄**

A2S/average_reward_1s�Ab��{,       ���E	8fm_��A��**

A2S/average_reward_1� �B��q,       ���E	=�|_��A��**

A2S/average_reward_1�:�A+��,       ���E	=p�_��AÆ**

A2S/average_reward_1�6BIߒ�,       ���E	�_�_��A��**

A2S/average_reward_1
�+B�T�j,       ���E	���_��A��**

A2S/average_reward_1� SC?
��,       ���E	v&=_��A��**

A2S/average_reward_1T[+D�P%,       ���E	�7�_��A��**

A2S/average_reward_1=�+D�(�%,       ���E	���_��A�**

A2S/average_reward_1�P|B��}�,       ���E	Z]�_��A��**

A2S/average_reward_1:ȥC�,�,       ���E	.|-_��Aѡ**

A2S/average_reward_1b��C�6k-,       ���E	�L9_��A�**

A2S/average_reward_1��UA4�l0,       ���E	�W_��AǢ**

A2S/average_reward_1NcB�e]�,       ���E	� �_��A��**

A2S/average_reward_1P�{C���,       ���E	S�_��A̦**

A2S/average_reward_1���B��l�,       ���E	ߠ�_��A��**

A2S/average_reward_1�P�C��W,       ���E	�g�_��AҪ**

A2S/average_reward_1#��Ar��,       ���E	�]_��A��**

A2S/average_reward_1V�C�Xm�,       ���E	њ_��Aʬ**

A2S/average_reward_1- A�eU�,       ���E	�0%_��A�**

A2S/average_reward_1�B� �,       ���E	N�9_��A��**

A2S/average_reward_1��8B�9�,       ���E	RԒ_��A��**

A2S/average_reward_1k�/DVe�g,       ���E	�	�_��A�**

A2S/average_reward_1B��C��,$,       ���E	V��_��A�**

A2S/average_reward_1j��A�Y��,       ���E	+]�_��A��**

A2S/average_reward_1`�Aȏ!\,       ���E	�0�_��AĻ**

A2S/average_reward_1�Bʙ+,       ���E	L'�_��A��**

A2S/average_reward_1��fB[�Ѱ,       ���E	��_��A��**

A2S/average_reward_1#:�C��T|,       ���E	�}�_��A��**

A2S/average_reward_11ɸBp�l,       ���E	��O_��A��**

A2S/average_reward_1M�-D��,       ���E	AZj_��A��**

A2S/average_reward_1Ӊ�B��,       ���E	;��_��A��**

A2S/average_reward_1��xBi*t�,       ���E	�1�_��A��**

A2S/average_reward_1S��C�[�,       ���E	`�_��A��**

A2S/average_reward_15A���,       ���E	�r�_��A��**

A2S/average_reward_1m�XC��,       ���E	�[_��A��**

A2S/average_reward_1#ȰA�5,       ���E	8�R_��A��**

A2S/average_reward_1<f)D-y�,       ���E	�ܬ_��A��**

A2S/average_reward_1�3Cd���,       ���E	ؐ!_��A��**

A2S/average_reward_1!,D�4{�,       ���E	�
`"_��A��**

A2S/average_reward_1��,D:��,       ���E	�4#_��A��**

A2S/average_reward_1��C��L�,       ���E	�C#_��A��**

A2S/average_reward_1Wp B�gt=x       ��!�	"/u._��A��**i

A2S/kl���;

A2S/policy_network_loss5q߿

A2S/value_network_loss�JPC

A2S/q_network_loss�TC�;�	,       ���E	�/�._��A��**

A2S/average_reward_1G�C�dک,       ���E	U�._��A��**

A2S/average_reward_1�B���,       ���E	�l�._��A��**

A2S/average_reward_1g�Aނ*,       ���E	�E�._��A��**

A2S/average_reward_1��B�]P1,       ���E	/_��A��**

A2S/average_reward_1iEB4�,       ���E	�L9/_��A��**

A2S/average_reward_1�Cf�',       ���E	nmC/_��A��**

A2S/average_reward_1���A�Ҁ�,       ���E	�!K/_��A��**

A2S/average_reward_1NH�AW�@�,       ���E	�`/_��A��**

A2S/average_reward_16��A����,       ���E	��/_��A��**

A2S/average_reward_1�B%���,       ���E	S��/_��A��**

A2S/average_reward_1]��Ay��*,       ���E	��/_��A��**

A2S/average_reward_1!iB�Sֿ,       ���E	V��/_��A��**

A2S/average_reward_1��8B�)1,       ���E	���/_��A��**

A2S/average_reward_1n)�A���,       ���E	%��/_��A��**

A2S/average_reward_1װ�B� K},       ���E	)��/_��A��**

A2S/average_reward_1�+�A:ٓ�,       ���E	�
0_��A��**

A2S/average_reward_1b@YAҴ�,       ���E	0_��A��**

A2S/average_reward_1��A:~7,       ���E	�50_��A��**

A2S/average_reward_1D�BR�r�,       ���E	�~C0_��A��**

A2S/average_reward_1{QA\��I,       ���E	V[�0_��A��**

A2S/average_reward_1%RzC`��,       ���E	���0_��A��**

A2S/average_reward_1��eB�Z9�,       ���E	�D�0_��A��**

A2S/average_reward_1U�@HJh�,       ���E	���0_��A��**

A2S/average_reward_1b�hB��,       ���E	f�1_��A��+*

A2S/average_reward_1�dqB+���,       ���E	kY1_��AЀ+*

A2S/average_reward_1o�Bڴ��,       ���E	&1_��A��+*

A2S/average_reward_1ԝBV��\,       ���E	A�M1_��A�+*

A2S/average_reward_1��BR��,       ���E	~^1_��A��+*

A2S/average_reward_1�?rA�W�X,       ���E	��1_��A��+*

A2S/average_reward_1�6�B�1`,       ���E	T��1_��Aу+*

A2S/average_reward_1w�B_��,       ���E	v 2_��A��+*

A2S/average_reward_1�^CB��[,       ���E	02_��A��+*

A2S/average_reward_1AؽB��/,       ���E	}�:2_��A��+*

A2S/average_reward_1"�A�:�&,       ���E	�K2_��A׆+*

A2S/average_reward_1:��Au�Z�,       ���E	*.]2_��A��+*

A2S/average_reward_1͢Ap�N,       ���E	S݃2_��A��+*

A2S/average_reward_1Gi�B��,       ���E	#&�2_��A҈+*

A2S/average_reward_1���B���,       ���E	7��2_��A��+*

A2S/average_reward_1Y�AĜ�>,       ���E	00�2_��A��+*

A2S/average_reward_1`\�B�@,       ���E	�8�2_��A��+*

A2S/average_reward_1G�XA�q܄,       ���E	�kB3_��A��+*

A2S/average_reward_1�AC�v�X,       ���E	�VL3_��A��+*

A2S/average_reward_1��AH�+2,       ���E	�sf3_��A�+*

A2S/average_reward_1���B��@�,       ���E	A{3_��A��+*

A2S/average_reward_1/�BZ�,       ���E	ܷ�3_��A��+*

A2S/average_reward_1���B��1�,       ���E	�%&4_��A��+*

A2S/average_reward_1���CzE�K,       ���E	E+4_��Aؒ+*

A2S/average_reward_1z�A���,       ���E	�rK4_��A�+*

A2S/average_reward_1���B�N
,       ���E	i�[4_��A��+*

A2S/average_reward_1>rB;�
,       ���E	��l4_��AӔ+*

A2S/average_reward_1WA�	�,       ���E	��4_��A��+*

A2S/average_reward_1�)�AzZO,       ���E		��4_��A��+*

A2S/average_reward_1I�B�Eg,       ���E	��4_��Aږ+*

A2S/average_reward_1rV�A$醙,       ���E	�z>6_��A+*

A2S/average_reward_1c5-D j��,       ���E	��Q6_��A��+*

A2S/average_reward_1mB5xSL,       ���E	�_6_��A��+*

A2S/average_reward_1Y��A�G�d,       ���E	۽s6_��A��+*

A2S/average_reward_1�X!B)k,       ���E	렂6_��A��+*

A2S/average_reward_1ɡB �-,       ���E	LL�6_��A��+*

A2S/average_reward_1�|�A�4B�,       ���E	�ܦ6_��A��+*

A2S/average_reward_1G�BQ$�,       ���E	�g�6_��A��+*

A2S/average_reward_1Ȫ�Aܳ��,       ���E	��'8_��A��+*

A2S/average_reward_1��+DU��f,       ���E	�GE8_��A��+*

A2S/average_reward_1�pBz0�,       ���E	I�59_��AЮ+*

A2S/average_reward_1�c�C�N��,       ���E	?bh9_��A��+*

A2S/average_reward_1<R�B�Ƣ�,       ���E	>):_��A��+*

A2S/average_reward_1�,�CMst�,       ���E	��\;_��A��+*

A2S/average_reward_1�&/D�0�D,       ���E	os�<_��A��+*

A2S/average_reward_1�qD��,       ���E	ɣ�=_��A��+*

A2S/average_reward_1�S�C?���,       ���E	��=_��A��+*

A2S/average_reward_1@hHB�C ,       ���E	�7?_��A��+*

A2S/average_reward_1n�*D��h`,       ���E	��N?_��A��+*

A2S/average_reward_1��cB\��Q,       ���E	�Is?_��A��+*

A2S/average_reward_1��Bl\�,       ���E	3�?_��A��+*

A2S/average_reward_1�mC�^�,       ���E	1<@_��A��+*

A2S/average_reward_1��C��#,       ���E	 ��@_��A��+*

A2S/average_reward_1��wC�!R,       ���E	���@_��A��+*

A2S/average_reward_1	|�B�-c�,       ���E	i_�A_��A��+*

A2S/average_reward_1���C%͢�,       ���E	�WC_��A��+*

A2S/average_reward_1�/D� Z+,       ���E	��C_��A��+*

A2S/average_reward_1凜A���x,       ���E	CVC_��A��+*

A2S/average_reward_1���B�b�,       ���E	�k�D_��A��+*

A2S/average_reward_1?1,D
�t,       ���E	!�D_��A��+*

A2S/average_reward_1��GB�q�,       ���E	���E_��A��+*

A2S/average_reward_1qP+D:�+�,       ���E	s'VG_��A��+*

A2S/average_reward_18�$DX���,       ���E	N��G_��A̂,*

A2S/average_reward_1S�CK>��,       ���E	7'H_��A�,*

A2S/average_reward_1��A~��^,       ���E	�-H_��A��,*

A2S/average_reward_1�e�B^��,       ���E	�"=H_��A؃,*

A2S/average_reward_1�0B���,       ���E	�MH_��A��,*

A2S/average_reward_1���A���K,       ���E	SH_��A��,*

A2S/average_reward_1Y�A�<�,       ���E	Sf&I_��Aو,*

A2S/average_reward_1�:�C)��,       ���E	26I_��A��,*

A2S/average_reward_1v*�A��1,       ���E	�jHI_��A͉,*

A2S/average_reward_1z3B�e�O,       ���E	L=QI_��A�,*

A2S/average_reward_1خA<���,       ���E	�3J_��A��,*

A2S/average_reward_1�W�C���,       ���E	�l�K_��A�,*

A2S/average_reward_1+C,D�,�',       ���E	�,�K_��A�,*

A2S/average_reward_1[ГB!N��,       ���E	�p�K_��A��,*

A2S/average_reward_1��B��`�,       ���E	�^L_��A͛,*

A2S/average_reward_1=��C4j�,       ���E	���M_��A��,*

A2S/average_reward_1��)D��z,       ���E	�2�M_��A�,*

A2S/average_reward_1��-B[d��,       ���E	5��M_��AΤ,*

A2S/average_reward_1�uB��z�,       ���E	�	N_��A��,*

A2S/average_reward_1l�8B'�,       ���E	��N_��A��,*

A2S/average_reward_1��Aw�,       ���E	JŀN_��A�,*

A2S/average_reward_1%=XC+�X�,       ���E	���N_��A��,*

A2S/average_reward_1c�BC�+m},       ���E	�F�N_��A��,*

A2S/average_reward_1��AH��,       ���E	���N_��Aͪ,*

A2S/average_reward_1Rd%B� �,       ���E	ؗ�O_��A��,*

A2S/average_reward_1^
�C��,       ���E	�  P_��A��,*

A2S/average_reward_1��vC;�/@,       ���E	��zQ_��Aܸ,*

A2S/average_reward_1te&D�^�,       ���E	��Q_��Aݹ,*

A2S/average_reward_1�f�B\��?,       ���E	F �Q_��A��,*

A2S/average_reward_1΁B���,       ���E	B�Q_��A̺,*

A2S/average_reward_17�DB��Y�,       ���E	���Q_��A�,*

A2S/average_reward_1���AU9�,       ���E	MS_��A��,*

A2S/average_reward_1,+DY��,       ���E	050T_��A��,*

A2S/average_reward_1��C��e,       ���E	q�AT_��A��,*

A2S/average_reward_1�r BȄ�z,       ���E	_�vT_��A��,*

A2S/average_reward_1�a�B��s�,       ���E	�ۓT_��A��,*

A2S/average_reward_1�	�By7.�,       ���E	~J�T_��A��,*

A2S/average_reward_1�C��H=,       ���E	"��T_��A��,*

A2S/average_reward_1O� B��%,       ���E	X U_��A��,*

A2S/average_reward_1QKBh� f,       ���E	X�.U_��A��,*

A2S/average_reward_1ֽ�A0��,       ���E	hq>U_��A��,*

A2S/average_reward_1ދ�A��,       ���E	��MU_��A��,*

A2S/average_reward_1�I�A�(,       ���E	M�cU_��A��,*

A2S/average_reward_1fFB�U��,       ���E	��U_��A��,*

A2S/average_reward_1�2GC��w�,       ���E	���V_��A��,*

A2S/average_reward_1��Dw�^,       ���E	H_�V_��A��,*

A2S/average_reward_1�h�A�.x*,       ���E	�BW_��A��,*

A2S/average_reward_1�z C��֕,       ���E	A;UW_��A��,*

A2S/average_reward_1�&
B���,       ���E	�zdW_��A��,*

A2S/average_reward_1K�B ��,       ���E	��X_��A��,*

A2S/average_reward_1ژ/D2�vl,       ���E	!��X_��A��,*

A2S/average_reward_1`Z$B�C�,       ���E	���X_��A��,*

A2S/average_reward_1?f(B���,       ���E	d�X_��A��,*

A2S/average_reward_1��A�j�(,       ���E	�Y_��A��,*

A2S/average_reward_1%��BS��r,       ���E	�g)Y_��A��,*

A2S/average_reward_1��A;���,       ���E	�];Y_��A��,*

A2S/average_reward_1�
B]��%,       ���E	��GY_��A��,*

A2S/average_reward_1Za�A.��,       ���E	�ĐY_��A��,*

A2S/average_reward_1��$Cp�9j,       ���E	� �Z_��A��,*

A2S/average_reward_1@j)D�c,       ���E	TR�[_��A��,*

A2S/average_reward_1��C,��,       ���E	[��[_��A��,*

A2S/average_reward_1?��A�9��x       ��!�	�>-f_��A��,*i

A2S/klu��:

A2S/policy_network_loss4� �

A2S/value_network_loss��TC

A2S/q_network_lossV�YC#ƀ�,       ���E	�Jlf_��A��,*

A2S/average_reward_1NCWr5�,       ���E	��xf_��A��,*

A2S/average_reward_1�z�A8;�,       ���E	���f_��A��,*

A2S/average_reward_1.�Cؚr0,       ���E	wN�f_��A��,*

A2S/average_reward_1��BLhv,       ���E	#�f_��A��,*

A2S/average_reward_1#�oA8�,       ���E	��Bg_��A��,*

A2S/average_reward_1H�C���,       ���E	KPg_��A��,*

A2S/average_reward_163�A�}Զ,       ���E	�'Vg_��A��,*

A2S/average_reward_1��A꫞3,       ���E	E�ag_��A��,*

A2S/average_reward_1�0�A�.�w,       ���E	��ng_��A��,*

A2S/average_reward_1��A���,       ���E	�Xyg_��A��,*

A2S/average_reward_1��
B�,�A,       ���E	��h_��A��-*

A2S/average_reward_1��,Dj�S�,       ���E	��i_��A��-*

A2S/average_reward_1�|C$�-�,       ���E	�|�i_��AՈ-*

A2S/average_reward_1�G�C+P�C,       ���E	F��i_��A��-*

A2S/average_reward_1�(B[x�,       ���E	�g�i_��A��-*

A2S/average_reward_1���A[9�7,       ���E	��j_��A��-*

A2S/average_reward_1V��CJ,'�,       ���E	���j_��A��-*

A2S/average_reward_1� �A'�~f,       ���E	D�k_��Aۏ-*

A2S/average_reward_1A��B�y��,       ���E	�wk_��A��-*

A2S/average_reward_1�Q)B�S�,       ���E	')k_��A��-*

A2S/average_reward_1���A�7:,       ���E	�قl_��A��-*

A2S/average_reward_1�7D�x~�,       ���E	��l_��A��-*

A2S/average_reward_1&�WA,;�=,       ���E	U��m_��A��-*

A2S/average_reward_1ڪ1Dp��,,       ���E	�n_��A��-*

A2S/average_reward_1F'�AI-��,       ���E	�n_��A�-*

A2S/average_reward_1
D;Bs8��,       ���E	�q=n_��Aڡ-*

A2S/average_reward_1\��Br��H,       ���E	�jOn_��A��-*

A2S/average_reward_1+B(Ӹ},       ���E	���n_��A��-*

A2S/average_reward_1C��Cэ<�,       ���E	:�o_��A��-*

A2S/average_reward_1T C�y�,       ���E	)��o_��A��-*

A2S/average_reward_1S�xC\�`�,       ���E	q�o_��A��-*

A2S/average_reward_1��A~B�,       ���E	�!�o_��A�-*

A2S/average_reward_1��BB��
	,       ���E	׹o_��A��-*

A2S/average_reward_1,�B,Y�7,       ���E	^��o_��A��-*

A2S/average_reward_1X�<A�N,       ���E	��o_��AЬ-*

A2S/average_reward_1-\C�`5,       ���E	�p_��A��-*

A2S/average_reward_1@üB�܊,       ���E	��Hp_��Aد-*

A2S/average_reward_1C;C���L,       ���E	�Mp_��A�-*

A2S/average_reward_1�ЬA둱[,       ���E	J�Sp_��A��-*

A2S/average_reward_1�7�A�7|�,       ���E	���p_��A��-*

A2S/average_reward_1a��C@��[,       ���E	;q_��Aȵ-*

A2S/average_reward_1e	Cz��<,       ���E	��hq_��Aȷ-*

A2S/average_reward_1��?C#�(�,       ���E	MH�q_��A��-*

A2S/average_reward_1dԝC����,       ���E	�6�r_��A�-*

A2S/average_reward_1��C{t��,       ���E	%l�r_��A��-*

A2S/average_reward_1k�B��F,       ���E	J�s_��A��-*

A2S/average_reward_1�B���,       ���E	.s_��A��-*

A2S/average_reward_1���A3�<m,       ���E	�A#s_��A��-*

A2S/average_reward_1I�@B�y�N,       ���E	Dd�s_��A��-*

A2S/average_reward_1N�vC}��[x       ��!�	c�k}_��A��-*i

A2S/kl���9

A2S/policy_network_loss��

A2S/value_network_loss�}C

A2S/q_network_loss�j~C����,       ���E	Ż�}_��A��-*

A2S/average_reward_1��C%�Ջ,       ���E	�s�}_��A��-*

A2S/average_reward_1�B��S,       ���E	 �~_��A��-*

A2S/average_reward_1��Ah�,       ���E	}5X~_��A��-*

A2S/average_reward_1M2OC���,       ���E	���~_��A��-*

A2S/average_reward_1�C��m�,       ���E	��~_��A��-*

A2S/average_reward_1�6*B��,       ���E	q�_��A��-*

A2S/average_reward_1�(7D�,       ���E	��/�_��A��-*

A2S/average_reward_1Pq0D��sc,       ���E	.�:�_��A��-*

A2S/average_reward_1�BP��K,       ���E	r�L�_��A��-*

A2S/average_reward_1��B��f5,       ���E	낁_��A��-*

A2S/average_reward_1ҌCL��,       ���E	��҂_��A��-*

A2S/average_reward_1�/2D �x�,       ���E	��9�_��A��-*

A2S/average_reward_13�qC;%,       ���E	��s�_��A��-*

A2S/average_reward_1�vC���,       ���E	�ʃ_��A��-*

A2S/average_reward_1�t7Cp>��,       ���E	ZO�_��A��-*

A2S/average_reward_1���Cj�+�,       ���E	��c�_��A��-*

A2S/average_reward_1>/BLu� ,       ���E	���_��A��-*

A2S/average_reward_1B܄Bfo
E,       ���E	ݏ�_��A��-*

A2S/average_reward_1���AF�:,       ���E	�R�_��A��-*

A2S/average_reward_1��8D�-�L,       ���E	�(�_��A��-*

A2S/average_reward_1��gA�O�,       ���E	+(.�_��A��-*

A2S/average_reward_1��AI%�,       ���E	k�I�_��A��-*

A2S/average_reward_1��`B(��,       ���E	؟V�_��A��-*

A2S/average_reward_1*��>�y ,       ���E	�op�_��A��-*

A2S/average_reward_1_]XB^���,       ���E	��v�_��A��-*

A2S/average_reward_1�˴A+�%,       ���E	�eȆ_��A��-*

A2S/average_reward_1�&	C��,       ���E	G�ۆ_��A��-*

A2S/average_reward_1w�B��],       ���E	Re�_��A��-*

A2S/average_reward_1+��B-���,       ���E	#l!�_��A��-*

A2S/average_reward_1Fr�A-y��,       ���E	�4�_��A��-*

A2S/average_reward_1�^Bn���,       ���E	��{�_��A��-*

A2S/average_reward_1G�C·��,       ���E	���_��A��-*

A2S/average_reward_1[�^Bj���,       ���E	2ߕ�_��A��-*

A2S/average_reward_1��BF��,       ���E	����_��A��-*

A2S/average_reward_1WC-~PZ,       ���E	���_��A��.*

A2S/average_reward_1{�B	%�~,       ���E	ܧ$�_��A��.*

A2S/average_reward_1#�Bsu�e,       ���E	 `��_��A��.*

A2S/average_reward_1#ىC��g,       ���E	5ͤ�_��A��.*

A2S/average_reward_1��jBH�Z�,       ���E	W���_��A܃.*

A2S/average_reward_1�B��,       ���E	��ƈ_��A��.*

A2S/average_reward_1��UB�,       ���E	� ݈_��AՄ.*

A2S/average_reward_1��Bp��,       ���E	��_��A��.*

A2S/average_reward_1��DB�Z��,       ���E	#^7�_��A��.*

A2S/average_reward_1دC��#�,       ���E	&��_��A��.*

A2S/average_reward_1�O;DUl��,       ���E	����_��A��.*

A2S/average_reward_1$�B��D,       ���E	�Ɋ_��A��.*

A2S/average_reward_1@�A6��,       ���E	���_��Aɐ.*

A2S/average_reward_1� Cl�׆,       ���E	��i�_��A��.*

A2S/average_reward_1;(C�I��,       ���E	�\~�_��A�.*

A2S/average_reward_1.�MB"���x       ��!�	_C�_��A�.*i

A2S/kl���:

A2S/policy_network_loss�U0�

A2S/value_network_lossA6�C

A2S/q_network_loss���C����,       ���E	�?r�_��A�.*

A2S/average_reward_1���B
��*,       ���E	.�͕_��A��.*

A2S/average_reward_1�1aCH"N,       ���E	�Eە_��A��.*

A2S/average_reward_1eDBf-�,       ���E	�`�_��AՖ.*

A2S/average_reward_1��BҮŉ,       ���E	�p�_��A��.*

A2S/average_reward_1UD%�0,       ���E	�z�_��A��.*

A2S/average_reward_1�B.jvT,       ���E	�1�_��A��.*

A2S/average_reward_1RjC�o�,       ���E	�WZ�_��A�.*

A2S/average_reward_1���B��,       ���E	�\�_��A��.*

A2S/average_reward_1VDW�%�,       ���E	�-j�_��A��.*

A2S/average_reward_10�B�;�W,       ���E	����_��Aަ.*

A2S/average_reward_1��A�Eaj,       ���E	�sØ_��A��.*

A2S/average_reward_1��C�ծ�,       ���E	3��_��Aͩ.*

A2S/average_reward_1�y2C����,       ���E	�*�_��A��.*

A2S/average_reward_1:�&A��1K,       ���E	��7�_��A��.*

A2S/average_reward_1�pB�
��,       ���E	�f�_��A��.*

A2S/average_reward_1���Br#�,       ���E	���_��A��.*

A2S/average_reward_1�Q�BΊ��,       ���E	f��_��Aˬ.*

A2S/average_reward_1�1�A���g,       ���E	_�љ_��A��.*

A2S/average_reward_1���B�η#,       ���E	g���_��A��.*

A2S/average_reward_1_C�B�.,       ���E	��_��A��.*

A2S/average_reward_1ԶB�q,       ���E	'"�_��A�.*

A2S/average_reward_1j�A�*3P,       ���E	�4�_��A��.*

A2S/average_reward_1	~B8v��,       ���E	'��_��A��.*

A2S/average_reward_1��CgJ2�,       ���E	�r�_��A��.*

A2S/average_reward_1{��B|9��,       ���E	@"�_��A��.*

A2S/average_reward_1�F�B�Kp�,       ���E	$�/�_��Aڴ.*

A2S/average_reward_1n��A�J�^,       ���E	�_�_��Aܵ.*

A2S/average_reward_1��B�֢3,       ���E	����_��A�.*

A2S/average_reward_1K�B���,       ���E	3ӛ_��A��.*

A2S/average_reward_1^CR^�,       ���E	!�_��A��.*

A2S/average_reward_1���B�E�,       ���E	l�_��Aݹ.*

A2S/average_reward_1�B�lh�,       ���E	�!�_��A��.*

A2S/average_reward_1So�A�u�|,       ���E	<�T�_��A�.*

A2S/average_reward_1�6�BN���,       ���E	��~�_��A��.*

A2S/average_reward_1�]�B��R9,       ���E	8��_��A׾.*

A2S/average_reward_1Q�CZԩ,       ���E	x�d�_��A��.*

A2S/average_reward_11�[C�?�,       ���E	���_��A��.*

A2S/average_reward_1IB�C���U,       ���E	_X%�_��A��.*

A2S/average_reward_1Gi�B��dw,       ���E	��>�_��A��.*

A2S/average_reward_1�~�AF�T�,       ���E	?��_��A��.*

A2S/average_reward_1��C�}]n,       ���E	)���_��A��.*

A2S/average_reward_1��B��԰,       ���E	����_��A��.*

A2S/average_reward_1{�@���,       ���E	a���_��A��.*

A2S/average_reward_1���A{� �,       ���E	��_��A��.*

A2S/average_reward_1^�Bu,       ���E	�_��A��.*

A2S/average_reward_1���B8h��,       ���E	!�M�_��A��.*

A2S/average_reward_1#�B9���,       ���E	��y�_��A��.*

A2S/average_reward_1��B X�,       ���E	ᒟ_��A��.*

A2S/average_reward_1��OBޛ��,       ���E	TXӟ_��A��.*

A2S/average_reward_1��B@���x       ��!�	��C�_��A��.*i

A2S/kl�/�:

A2S/policy_network_loss|�.�

A2S/value_network_lossNKC

A2S/q_network_lossRPC�ӭ,       ���E	��Z�_��A��.*

A2S/average_reward_1T��A��0,       ���E	׷�_��A��.*

A2S/average_reward_1�:C�)R;,       ���E	Uϩ_��A��.*

A2S/average_reward_1E�B��w�,       ���E	@{ݩ_��A��.*

A2S/average_reward_1�� Bz�I,       ���E	z)<�_��A��.*

A2S/average_reward_1p�KC"���,       ���E	 �V�_��A��.*

A2S/average_reward_10n�A6f��,       ���E	�L{�_��A��.*

A2S/average_reward_1��B����,       ���E	?���_��A��.*

A2S/average_reward_1��|B��<,       ���E	1�Ѫ_��A��.*

A2S/average_reward_1|�B�@�2,       ���E	���_��A��.*

A2S/average_reward_1G�BVoT�,       ���E	��_��A��.*

A2S/average_reward_1K�B�M�,       ���E	+3-�_��A��.*

A2S/average_reward_1DPNB��d,       ���E	��?�_��A��.*

A2S/average_reward_1�}A�,+�,       ���E	���_��A��.*

A2S/average_reward_1k�VC�,       ���E	���_��A��.*

A2S/average_reward_1���Cz^�,       ���E	�<�_��A��.*

A2S/average_reward_1g��Bk�e�,       ���E	�"K�_��A��.*

A2S/average_reward_1��^Ah�A,       ���E	�Î�_��A��.*

A2S/average_reward_1�5C&�,       ���E	�o�_��A��.*

A2S/average_reward_1�E�C)��,       ���E	�i�_��A��.*

A2S/average_reward_1�ɂC���I,       ���E	�];�_��A��.*

A2S/average_reward_13RMCK��|,       ���E	axX�_��A��.*

A2S/average_reward_1��B3w�,       ���E	�Km�_��A��.*

A2S/average_reward_1��0B=ӡ,       ���E	��}�_��A��.*

A2S/average_reward_1^�WB
��;,       ���E	�)ڮ_��A��.*

A2S/average_reward_1��vC!
!,       ���E	%�_��A��.*

A2S/average_reward_1^JA��y,       ���E	K���_��A��.*

A2S/average_reward_1�KBmT,       ���E	�3��_��A��.*

A2S/average_reward_1z��A�+�,       ���E	�o�_��A��.*

A2S/average_reward_1n�BPl�,       ���E	�x$�_��A��.*

A2S/average_reward_1\��Bt[Sq,       ���E	+�+�_��A��.*

A2S/average_reward_1�yB�ϝ,       ���E	i#6�_��A��.*

A2S/average_reward_1�6B{[�w,       ���E	C���_��A��.*

A2S/average_reward_1�Y�C����,       ���E	����_��A��.*

A2S/average_reward_1#B�v�,       ���E	�į_��A��.*

A2S/average_reward_1GBH��,       ���E	�J�_��A��.*

A2S/average_reward_1 ��C�r�$,       ���E	���_��A��.*

A2S/average_reward_1T��C�t�0,       ���E	8��_��A��.*

A2S/average_reward_1jE�A��,       ���E	."�_��A��.*

A2S/average_reward_1���A���,       ���E	�*#�_��A��.*

A2S/average_reward_1�A���,       ���E	��E�_��A��.*

A2S/average_reward_1�)�B.Ӌ�,       ���E	+�^�_��AȂ/*

A2S/average_reward_1�oDs�8�,       ���E	�xe�_��A؂/*

A2S/average_reward_1��AW96�,       ���E	{��_��A��/*

A2S/average_reward_1)��C^(ϲ,       ���E	�5'�_��A�/*

A2S/average_reward_12�B�~�,       ���E	�,��_��A��/*

A2S/average_reward_1�	�C���,       ���E	�L��_��Aщ/*

A2S/average_reward_1$�B�X,       ���E	����_��A��/*

A2S/average_reward_1��Brrq,       ���E	��E�_��A��/*

A2S/average_reward_1[�C�3To,       ���E	��g�_��A��/*

A2S/average_reward_1�t�BV��x       ��!�	=
��_��A��/*i

A2S/kl�Q�:

A2S/policy_network_lossh ;�

A2S/value_network_lossS:�C

A2S/q_network_loss�3�C��b,       ���E	o���_��AƎ/*

A2S/average_reward_1V��Aԯk,       ���E	���_��A�/*

A2S/average_reward_1j�B]R,       ���E	��_��A��/*

A2S/average_reward_1Lj�A� d,       ���E	{ƥ�_��A��/*

A2S/average_reward_1%b�CDer�,       ���E	�u��_��Aǒ/*

A2S/average_reward_1�LB=�7H,       ���E	����_��Aܒ/*

A2S/average_reward_1sx�A��k�,       ���E	���_��Aȓ/*

A2S/average_reward_1�s�B>�c�,       ���E	.C�_��A�/*

A2S/average_reward_1|��Ap��,       ���E	N��_��A��/*

A2S/average_reward_1�vA;��,       ���E	��_��A��/*

A2S/average_reward_1�68Bĥ:�,       ���E	94A�_��A��/*

A2S/average_reward_1G\�B�_�,       ���E	��z�_��A��/*

A2S/average_reward_13�BU�E�,       ���E	5#��_��AŖ/*

A2S/average_reward_1�7�Aܳq,       ���E	l��_��Aۖ/*

A2S/average_reward_18Z�AB�T,       ���E	�_��_��Aڗ/*

A2S/average_reward_1r��BA��8,       ���E	��ݿ_��A�/*

A2S/average_reward_1%�B!��,       ���E	�_��A��/*

A2S/average_reward_1Tg�A�\�A,       ���E	��_��A��/*

A2S/average_reward_1���AD�|,       ���E	���_��A��/*

A2S/average_reward_1�C�A
�bR,       ���E	����_��AӞ/*

A2S/average_reward_1y+ D�W��,       ���E	�O��_��A�/*

A2S/average_reward_1���A���{,       ���E	���_��A��/*

A2S/average_reward_1۵Bi/R~,       ���E	�n��_��A��/*

A2S/average_reward_1L�C̟�H,       ���E	M��_��A��/*

A2S/average_reward_1��fC,F,       ���E	t@$�_��AΥ/*

A2S/average_reward_18,�A�O��,       ���E	�02�_��A��/*

A2S/average_reward_1��.B��C�,       ���E	�Q=�_��A��/*

A2S/average_reward_1$B��Bl,       ���E	��D�_��A��/*

A2S/average_reward_1�u�AC��,       ���E	�+l�_��A��/*

A2S/average_reward_1A�B��z,       ���E	���_��A��/*

A2S/average_reward_16��B�ĺ,       ���E	�l��_��A��/*

A2S/average_reward_1s�Ca,�3,       ���E	Hg��_��AŪ/*

A2S/average_reward_1�)�B���,       ���E	2��_��A��/*

A2S/average_reward_1��Bk���,       ���E	xs�_��A��/*

A2S/average_reward_1E@�CO�ֶ,       ���E	'2	�_��A��/*

A2S/average_reward_1��Ci%�,       ���E	�x�_��A��/*

A2S/average_reward_1��$BBY�,       ���E	���_��A��/*

A2S/average_reward_1z��C���,       ���E	����_��A��/*

A2S/average_reward_1�i�Bt���,       ���E	����_��Aص/*

A2S/average_reward_1QBM��G,       ���E	���_��A��/*

A2S/average_reward_1��FBk_�,       ���E	�]��_��A��/*

A2S/average_reward_1*[�Ab���,       ���E	��_��Aٶ/*

A2S/average_reward_1��lBl,       ���E	��_��A��/*

A2S/average_reward_1G�]B�%�,       ���E	�bA�_��A��/*

A2S/average_reward_1�	 C�la,       ���E	+�L�_��A�/*

A2S/average_reward_1��B��j,       ���E	�IR�_��A��/*

A2S/average_reward_1��A�z��,       ���E	 �Z�_��A��/*

A2S/average_reward_1Ъ�A�G},       ���E	x���_��A�/*

A2S/average_reward_1��C�:�,       ���E	\��_��A��/*

A2S/average_reward_1d��A��n,       ���E	�62�_��A�/*

A2S/average_reward_1��C�Ϋ,       ���E	7���_��A��/*

A2S/average_reward_1j�/D��N,       ���E	�T��_��A��/*

A2S/average_reward_1M ,D�Q��,       ���E	���_��A��/*

A2S/average_reward_1{��A>�e,       ���E	%d�_��A��/*

A2S/average_reward_1��B��,       ���E	�C��_��A��/*

A2S/average_reward_1��pC�J��,       ���E	�P��_��A��/*

A2S/average_reward_1�%D��,       ���E	d��_��A��/*

A2S/average_reward_1��|C�+L[,       ���E	��_��A��/*

A2S/average_reward_1�7�C!g�u,       ���E	�߸�_��A��/*

A2S/average_reward_1��XB-��,       ���E	e���_��A��/*

A2S/average_reward_1�F�A�6m,       ���E	����_��A��/*

A2S/average_reward_1���A���~,       ���E	����_��A��/*

A2S/average_reward_1˙BX� m,       ���E	B��_��A��/*

A2S/average_reward_1F��A3�w,       ���E	?��_��A��/*

A2S/average_reward_1�~�Aj[,       ���E	��_��A��/*

A2S/average_reward_1r�C�Mù,       ���E	n��_��A��/*

A2S/average_reward_1�NB�oa�,       ���E	�/��_��A��/*

A2S/average_reward_16�ABÌ�,       ���E	�w�_��A��/*

A2S/average_reward_1�	C�,       ���E	c��_��A��/*

A2S/average_reward_1)��A{���,       ���E	�0'�_��A��/*

A2S/average_reward_1��B����,       ���E	�:�_��A��/*

A2S/average_reward_1JciB�*1,       ���E	g��_��A��/*

A2S/average_reward_1O D��,       ���E	W]��_��A��/*

A2S/average_reward_1��BT��,       ���E	�!��_��A��/*

A2S/average_reward_1X��A���J,       ���E	�%��_��A��/*

A2S/average_reward_1�3NB�b#,       ���E	I�_��A��/*

A2S/average_reward_1��BR/��,       ���E	R��_��A��/*

A2S/average_reward_1�QB��,       ���E	�=U�_��A��/*

A2S/average_reward_1���B�?�-,       ���E	�W^�_��A��/*

A2S/average_reward_1�� AT|+�,       ���E	XlA�_��A��/*

A2S/average_reward_1]��C{Ă�,       ���E	@�|�_��A��/*

A2S/average_reward_1(��B��},       ���E	�9��_��A��/*

A2S/average_reward_1�۪A� ��,       ���E	|��_��A��/*

A2S/average_reward_1�R%B�?`,       ���E	K���_��A��/*

A2S/average_reward_1�BA���,       ���E	�Gl�_��A��/*

A2S/average_reward_10�CS��,       ���E	��v�_��A��0*

A2S/average_reward_1�$�ABs�,       ���E	����_��A��0*

A2S/average_reward_1
��Aǔ�c,       ���E	��7�_��A��0*

A2S/average_reward_1k�C��k�,       ���E	�F�_��A��0*

A2S/average_reward_1	H�A!�,       ���E	�f�_��A��0*

A2S/average_reward_1}G&D*��,       ���E	w���_��A��0*

A2S/average_reward_1��C��,       ���E	S�_��A�0*

A2S/average_reward_1�(.D�̩,       ���E	�n�_��A��0*

A2S/average_reward_1*/-B�`��,       ���E	���_��A��0*

A2S/average_reward_1+"D�n�,       ���E	����_��A�0*

A2S/average_reward_1��/D��n�,       ���E	��_��A®0*

A2S/average_reward_1x�(D`�F�,       ���E	��=�_��A̯0*

A2S/average_reward_1�_�B�k�,       ���E	4�J�_��A��0*

A2S/average_reward_1��B&���,       ���E	Ė��_��A�0*

A2S/average_reward_1*-D��
W,       ���E	�A��_��A��0*

A2S/average_reward_11B��=�,       ���E	|<��_��A��0*

A2S/average_reward_1{-�BH��x       ��!�	Ҿ��_��A��0*i

A2S/kl�A8<

A2S/policy_network_loss>��

A2S/value_network_loss$JC

A2S/q_network_loss��NC_�G,       ���E	����_��A͹0*

A2S/average_reward_1�v�At�6f,       ���E	z��_��A��0*

A2S/average_reward_1��2B���z,       ���E	\a�_��A��0*

A2S/average_reward_1O�C<�,       ���E	С�_��A�0*

A2S/average_reward_1V�B�1�F,       ���E	�_$�_��A��0*

A2S/average_reward_1��B��}�,       ���E	�),�_��A��0*

A2S/average_reward_1_-�A�Í,       ���E	g�=�_��A��0*

A2S/average_reward_1��RB���4,       ���E	�Uv�_��A��0*

A2S/average_reward_1�JCM��,       ���E	n��_��A��0*

A2S/average_reward_1e}SCZF��,       ���E	+R��_��A��0*

A2S/average_reward_1�Q�AKe��,       ���E	�f��_��A��0*

A2S/average_reward_1��6D�bA`,       ���E	s�
�_��A��0*

A2S/average_reward_1�ZBW�,,       ���E	��_��A��0*

A2S/average_reward_1���A����,       ���E	iA�_��A��0*

A2S/average_reward_1%�C�Y>K,       ���E	IPK�_��A��0*

A2S/average_reward_1��B�D,       ���E	�8�_��A��0*

A2S/average_reward_1=�C��,       ���E	��_��A��0*

A2S/average_reward_1�4�ApOZ,       ���E	+�_��A��0*

A2S/average_reward_1�&�A���z,       ���E	l�_��A��0*

A2S/average_reward_1w(A֭�,       ���E	���_��A��0*

A2S/average_reward_1��yA�y�y,       ���E	28_�_��A��0*

A2S/average_reward_1��C��@�,       ���E	�j�_��A��0*

A2S/average_reward_1'�A뮰X,       ���E	zS�_��A��0*

A2S/average_reward_1}˒C�,       ���E	�H
�_��A��0*

A2S/average_reward_1K�A�\�,       ���E	��8�_��A��0*

A2S/average_reward_1G\�B���,       ���E	�?a�_��A��0*

A2S/average_reward_1�˧B��M,       ���E	���_��A��0*

A2S/average_reward_1�qrC�%�,       ���E	����_��A��0*

A2S/average_reward_1�'B���y,       ���E	U���_��A��0*

A2S/average_reward_1�!EB&#�^,       ���E	�E7�_��A��0*

A2S/average_reward_1�`C��l,       ���E	�N{�_��A��0*

A2S/average_reward_1���B=��,       ���E	�0��_��A��0*

A2S/average_reward_1	n�BRN�&,       ���E	���_��A��0*

A2S/average_reward_1��A�Stb,       ���E	�y��_��A��0*

A2S/average_reward_1X�B`�N�,       ���E	�u��_��A��0*

A2S/average_reward_1��A���5,       ���E	�^#�_��A��0*

A2S/average_reward_1��Cj�c,       ���E	�1�_��A��0*

A2S/average_reward_1�Dl&s,       ���E	��W�_��A��0*

A2S/average_reward_1^=�B�n,       ���E	0i�_��A��0*

A2S/average_reward_1�B<�v�,       ���E	颒�_��A��0*

A2S/average_reward_1h�B�f�,       ���E	�ˣ�_��A��0*

A2S/average_reward_1�C�@�V��,       ���E	�r��_��A��0*

A2S/average_reward_1K(CƦ��,       ���E	�t�_��A��0*

A2S/average_reward_1|�4D�q��,       ���E	��-�_��A��0*

A2S/average_reward_1H�B&�4a,       ���E	[�:�_��A��0*

A2S/average_reward_1\GA�}i�,       ���E	&���_��A��0*

A2S/average_reward_1�-C���!,       ���E	{G��_��A��0*

A2S/average_reward_1�rB��z,       ���E	���_��A��0*

A2S/average_reward_1 35D�S��,       ���E	�6��_��A��0*

A2S/average_reward_1(�B51o,       ���E	'O��_��A��1*

A2S/average_reward_1h��C�,       ���E	#�#�_��A�1*

A2S/average_reward_1*�7D0�Lx       ��!�	K{��_��A�1*i

A2S/kl�� :

A2S/policy_network_loss1��

A2S/value_network_loss�ׅC

A2S/q_network_loss%��C���,       ���E	y ��_��A��1*

A2S/average_reward_1&m�A��:�,       ���E	w��_��A��1*

A2S/average_reward_1�.�B�R,       ���E	\�)�_��A��1*

A2S/average_reward_1��B"-�R,       ���E	�J�_��A��1*

A2S/average_reward_1hU�B�[�$,       ���E	RXW�_��A��1*

A2S/average_reward_1x�1A�L)�,       ���E	��`�_��AЊ1*

A2S/average_reward_1O��A4+�X,       ���E	c/��_��A׋1*

A2S/average_reward_1�� Ct	3V,       ���E	�9��_��A��1*

A2S/average_reward_1��Bώ�,       ���E	�ڻ�_��Aߌ1*

A2S/average_reward_1B��B����,       ���E	�.��_��A��1*

A2S/average_reward_1�}�@�X--,       ���E	�B��_��A��1*

A2S/average_reward_1vZC�R�,       ���E	���_��A��1*

A2S/average_reward_1�)�A�IN,       ���E	o�?�_��A��1*

A2S/average_reward_1�"C]��,       ���E	�K�_��Aΐ1*

A2S/average_reward_1u�AA�#X�,       ���E	a_p�_��Aő1*

A2S/average_reward_1���B~�,       ���E	p��_��A��1*

A2S/average_reward_1�)B]�0,       ���E	�P��_��A��1*

A2S/average_reward_1��A��lj,       ���E	qs��_��A��1*

A2S/average_reward_1���BX�,       ���E	���_��A��1*

A2S/average_reward_1Q��C���P,       ���E	(�&�_��A��1*

A2S/average_reward_1�� BW$�,       ���E	�3�_��A��1*

A2S/average_reward_1�EB?��,       ���E	+_�_��AǗ1*

A2S/average_reward_1��CW�%,       ���E	Kf�_��Aݗ1*

A2S/average_reward_1R��A�H6^,       ���E	����_��A��1*

A2S/average_reward_1�+C�v_,       ���E	
Q��_��Aޚ1*

A2S/average_reward_1�C6&Y�,       ���E	���_��A��1*

A2S/average_reward_1z��A���,       ���E	(�� `��A��1*

A2S/average_reward_10q�Ci���,       ���E	7S�`��Aڥ1*

A2S/average_reward_1$u�C���,,       ���E	e�`��A�1*

A2S/average_reward_1z��BB3�,       ���E	�<`��AϮ1*

A2S/average_reward_1B)5D�I�,       ���E	�o�`��Aٱ1*

A2S/average_reward_1uB�C�LG.,       ���E	v��`��A��1*

A2S/average_reward_1�\�A�P��,       ���E	���`��A�1*

A2S/average_reward_1}<B^�Ϭ,       ���E	_ �`��A��1*

A2S/average_reward_1��vAl��,       ���E	8�`��A��1*

A2S/average_reward_1���A�P��,       ���E	X�`��A��1*

A2S/average_reward_1��wB�0�,       ���E	A�`��A�1*

A2S/average_reward_1�D�f�,       ���E	�%`��A��1*

A2S/average_reward_1&U�A���%,       ���E	'Z�`��A��1*

A2S/average_reward_1�`_Cy7�t,       ���E	7�`��A��1*

A2S/average_reward_1��BO<�X,       ���E	lП`��A��1*

A2S/average_reward_1eD��5�,       ���E	6D�`��A��1*

A2S/average_reward_1��C@�,`f,       ���E	�W<`��A��1*

A2S/average_reward_1�S�C:}��,       ���E	N>E`��A��1*

A2S/average_reward_1���@�5õ,       ���E	ZdN`��A��1*

A2S/average_reward_1ǋ�A�S��,       ���E	���`��A��1*

A2S/average_reward_1U�CѨ�A,       ���E	�k�`��A��1*

A2S/average_reward_1 ��A�W�,       ���E	�%`��A��1*

A2S/average_reward_1d��B�@�,       ���E	��,`��A��1*

A2S/average_reward_1,��A���,       ���E	K{�	`��A��1*

A2S/average_reward_15;DWd�,       ���E	���	`��A��1*

A2S/average_reward_1��C
Y�x       ��!�	��`��A��1*i

A2S/kl�ã:

A2S/policy_network_loss�<�

A2S/value_network_loss�wC

A2S/q_network_lossXF|C���,       ���E	�4`��A��1*

A2S/average_reward_1�Y�B˿a,       ���E	uqy`��A��1*

A2S/average_reward_17�C�!׾,       ���E	n�o`��A��1*

A2S/average_reward_1��D��l-,       ���E	Z�`��A��1*

A2S/average_reward_1ԀC�A�,       ���E	`��`��A��1*

A2S/average_reward_169�B�w��,       ���E	1`��A��1*

A2S/average_reward_1���BX:�,       ���E	6;`��A��1*

A2S/average_reward_1K�B�2{�,       ���E	�S`��A��1*

A2S/average_reward_1�G�A#�,       ���E	#hu`��A��1*

A2S/average_reward_1Xk3Bɐ۞,       ���E	uء`��A��1*

A2S/average_reward_1��Bs	�,       ���E	M�`��A��1*

A2S/average_reward_1V��C�}̈́,       ���E	�``��A��1*

A2S/average_reward_1���A�p��,       ���E	�
`��A��1*

A2S/average_reward_1X	�Aⴟ,       ���E	R�J`��A��1*

A2S/average_reward_1�	C�O�,       ���E	�'O`��A��1*

A2S/average_reward_1,ЙA7ï,       ���E	Te�`��A��1*

A2S/average_reward_1�8CS)2�,       ���E	��`��A��1*

A2S/average_reward_1L,B��6,       ���E	y�`��A��1*

A2S/average_reward_1�nC-!,       ���E	�;�`��A��1*

A2S/average_reward_1Z BDE,       ���E	��z`��A��1*

A2S/average_reward_1�#�C��x,       ���E	
��`��A��1*

A2S/average_reward_1���A��],       ���E	]��`��A��1*

A2S/average_reward_1���BQ�چ,       ���E	��`��A��1*

A2S/average_reward_1�kC*Z�,       ���E	(��`��A��1*

A2S/average_reward_1}D��j,       ���E	�i`��A��1*

A2S/average_reward_1{�B�/k.,       ���E	m�^`��A��1*

A2S/average_reward_1��TCQ�8�,       ���E	�n�`��A��1*

A2S/average_reward_1���BQ!D,       ���E	r�`��A��1*

A2S/average_reward_1�C(u1j,       ���E	M��`��A��1*

A2S/average_reward_1孧A��0�,       ���E	=T`��A��2*

A2S/average_reward_1��B=�@,       ���E	��"`��A��2*

A2S/average_reward_1�@Y�ג,       ���E	;``��A��2*

A2S/average_reward_1qC�Oh,       ���E	���`��A��2*

A2S/average_reward_1���B���`,       ���E	`�`��A̓2*

A2S/average_reward_1
��A�]�,       ���E	OH�`��A�2*

A2S/average_reward_11%BQG^�,       ���E	�D`��Aڋ2*

A2S/average_reward_1�;D&;��,       ���E	��`��A2*

A2S/average_reward_1�Q<D��,       ���E	/v�`��AϘ2*

A2S/average_reward_1���C���|,       ���E	���`��A�2*

A2S/average_reward_1��`Cs� �,       ���E	���`��A��2*

A2S/average_reward_1��B]7i�,       ���E	0'	 `��A��2*

A2S/average_reward_1Tw�AqE��,       ���E	�?L `��A��2*

A2S/average_reward_1�� C�O4,       ���E	0R `��A��2*

A2S/average_reward_1 ��A�5,       ���E	:�Z `��A��2*

A2S/average_reward_1r·A�S 1,       ���E	dj `��Aޝ2*

A2S/average_reward_1��@B+-�I,       ���E	��� `��AƟ2*

A2S/average_reward_1�X=C�e9,       ���E	je�!`��A��2*

A2S/average_reward_17#D�,       ���E	���!`��A��2*

A2S/average_reward_1�~�B~+P,       ���E	��!`��A�2*

A2S/average_reward_1�i*C"��,       ���E	|�"`��A֩2*

A2S/average_reward_1넽B;��,       ���E	jj$"`��A�2*

A2S/average_reward_1�$�AND�x       ��!�	[3,`��A�2*i

A2S/kl+$�:

A2S/policy_network_loss(;3�

A2S/value_network_loss,�WC

A2S/q_network_loss��ZC��k�,       ���E	�E,`��Aܪ2*

A2S/average_reward_1AĺBƁ\,       ���E	��,`��A��2*

A2S/average_reward_10��C�\H�,       ���E	�m3.`��A��2*

A2S/average_reward_1�z%D};�,       ���E	�o.`��A��2*

A2S/average_reward_1��BO�.,       ���E	�g�/`��A��2*

A2S/average_reward_1�C:Dۘ�~,       ���E	L0`��A�2*

A2S/average_reward_1'�(B��e,       ���E	�x20`��Aڿ2*

A2S/average_reward_1���B��i.,       ���E	<ي0`��A��2*

A2S/average_reward_1��ACͧ�,       ���E	��0`��A��2*

A2S/average_reward_1��A@&ߘ,       ���E	���0`��A��2*

A2S/average_reward_1B�BH���,       ���E	C�0`��A��2*

A2S/average_reward_1�YB�y��,       ���E	��0`��A��2*

A2S/average_reward_1��B	K�s,       ���E	�1`��A��2*

A2S/average_reward_1f��BI)e�,       ���E	 1`��A��2*

A2S/average_reward_1�W�A�m!,       ���E	�n/1`��A��2*

A2S/average_reward_17ԄAˉ�2,       ���E	#a2`��A��2*

A2S/average_reward_1���C�D��,       ���E	�H2`��A��2*

A2S/average_reward_1a�C��,       ���E	�~2`��A��2*

A2S/average_reward_1P��BT�B�,       ���E	e��2`��A��2*

A2S/average_reward_1�m�A�b6�,       ���E	+~�2`��A��2*

A2S/average_reward_1kYBB�,       ���E	l�2`��A��2*

A2S/average_reward_12ޖA��K,       ���E	�>23`��A��2*

A2S/average_reward_1��gC1@Z�,       ���E	�v^3`��A��2*

A2S/average_reward_1s�B�U&R,       ���E	ٸ3`��A��2*

A2S/average_reward_1��LC1w��,       ���E	��3`��A��2*

A2S/average_reward_1��Aǒ�C,       ���E	�~�3`��A��2*

A2S/average_reward_17��B�^
�,       ���E	�Or4`��A��2*

A2S/average_reward_1��C5v/�,       ���E	+k�4`��A��2*

A2S/average_reward_1$eTB0�v,       ���E	��4`��A��2*

A2S/average_reward_1�C�@��,       ���E	,��4`��A��2*

A2S/average_reward_1���A��,       ���E	X#�4`��A��2*

A2S/average_reward_1�k�B�1�,       ���E	���4`��A��2*

A2S/average_reward_1hY�B�$1,       ���E	��5`��A��2*

A2S/average_reward_1w��A�[��,       ���E	�n6`��A��2*

A2S/average_reward_1�p;D	7t,       ���E	~��6`��A��2*

A2S/average_reward_1�C��J,       ���E	���6`��A��2*

A2S/average_reward_1��B��� ,       ���E	/M@7`��A��2*

A2S/average_reward_1�q�C��`�,       ���E	�iw7`��A��2*

A2S/average_reward_1�CW���,       ���E	O��7`��A��2*

A2S/average_reward_1^��C�!�,       ���E	�R8`��A��2*

A2S/average_reward_1��BX���,       ���E	��|9`��A��2*

A2S/average_reward_1G�=D��[�,       ���E	j�9`��A��2*

A2S/average_reward_1�RC����,       ���E	���9`��A��2*

A2S/average_reward_1���An���,       ���E	~I�9`��A��2*

A2S/average_reward_1��BE�,       ���E	͹:`��A��2*

A2S/average_reward_1ݕ�B�M��,       ���E	�_>;`��A��2*

A2S/average_reward_1&�:D�WxE,       ���E	�"d;`��AҀ3*

A2S/average_reward_1+��B��u�,       ���E	�ʁ;`��A��3*

A2S/average_reward_1��B�+�,       ���E	�9<`��A��3*

A2S/average_reward_1AF�C\���,       ���E	-=?<`��A��3*

A2S/average_reward_1���9��,       ���E	�Z<`��A��3*

A2S/average_reward_1 Bx�VQx       ��!�	t[JF`��A��3*i

A2S/kl�ڸ:

A2S/policy_network_loss�$�

A2S/value_network_loss���C

A2S/q_network_loss���CNymx,       ���E	2�`F`��Aǆ3*

A2S/average_reward_1�]RB�D>,       ���E	?�iF`��A�3*

A2S/average_reward_1!�B��*�,       ���E	���F`��A��3*

A2S/average_reward_1�W~C/�V;,       ���E	�<�F`��A��3*

A2S/average_reward_1$=�Ab�f,       ���E	i��F`��A҉3*

A2S/average_reward_1��B�$�,       ���E	�dkG`��A�3*

A2S/average_reward_1��CCN��,       ���E	�xG`��A��3*

A2S/average_reward_1%��Ae94,       ���E	�5�G`��A��3*

A2S/average_reward_1^>�BC��,       ���E	���G`��A�3*

A2S/average_reward_1k��B��
?,       ���E	��TH`��A��3*

A2S/average_reward_1�,�C8��;,       ���E	��H`��A��3*

A2S/average_reward_1~m�B�/fw,       ���E	ڬOI`��A��3*

A2S/average_reward_1O��C$���,       ���E	�cJ`��Aʛ3*

A2S/average_reward_1� �C�>��,       ���E	�"HJ`��A��3*

A2S/average_reward_1�)�B��,       ���E	|SJ`��Aܜ3*

A2S/average_reward_1���A(�,       ���E	�~K`��Aã3*

A2S/average_reward_1I�(D��۹,       ���E	���K`��Aݣ3*

A2S/average_reward_1cbA�@�,       ���E	�l�K`��AХ3*

A2S/average_reward_18�IC,?�,       ���E	q��K`��A��3*

A2S/average_reward_1DB��ݢ,       ���E	E��K`��A��3*

A2S/average_reward_1�j�A���,       ���E	*�L`��Aۦ3*

A2S/average_reward_1�d\Bd��#,       ���E	�L`��A��3*

A2S/average_reward_1��BP��S,       ���E	��NL`��A��3*

A2S/average_reward_1�C�=Z,       ���E	�6TL`��AĨ3*

A2S/average_reward_1�ѡA�e�,       ���E	��aL`��A�3*

A2S/average_reward_1�#BW�,       ���E	��M`��A��3*

A2S/average_reward_1���C[=U�,       ���E	�M`��A��3*

A2S/average_reward_1�μA�Q�,       ���E	�(+M`��A��3*

A2S/average_reward_1-�B��Z�,       ���E	a��M`��A��3*

A2S/average_reward_1��C��;,       ���E	�f�M`��A��3*

A2S/average_reward_17j�A��U,       ���E	��KN`��A��3*

A2S/average_reward_1�!9C�:,       ���E	�WN`��A��3*

A2S/average_reward_1W��A���8,       ���E	�O`��A��3*

A2S/average_reward_1ǥ�C"��,       ���E	B�WO`��A�3*

A2S/average_reward_1(�+C�r��,       ���E	�SeO`��A��3*

A2S/average_reward_1FPB\,       ���E	-�nO`��A��3*

A2S/average_reward_1c�A}D�c,       ���E	�*~O`��A�3*

A2S/average_reward_14��A�T,       ���E	�k�O`��A��3*

A2S/average_reward_1��CGqH,       ���E	��O`��A��3*

A2S/average_reward_1 @�A9�,       ���E	���O`��A��3*

A2S/average_reward_1��Bw��,       ���E	gLP`��A۽3*

A2S/average_reward_1��B{.,       ���E	�x	P`��A�3*

A2S/average_reward_1���A��X�,       ���E	��P`��A��3*

A2S/average_reward_1b+B
;#�,       ���E	�އP`��A��3*

A2S/average_reward_1�b�Cą�,       ���E	��P`��A��3*

A2S/average_reward_1+&-A"�u,       ���E	��9Q`��A��3*

A2S/average_reward_1�ؼC�Y�,       ���E	6�Q`��A��3*

A2S/average_reward_1�=C�&��,       ���E	���Q`��A��3*

A2S/average_reward_1��"B槑(,       ���E	�H�Q`��A��3*

A2S/average_reward_1�L�A�E�,       ���E	W[�Q`��A��3*

A2S/average_reward_1}�A�PI�,       ���E	�[R`��A��3*

A2S/average_reward_1�֤C\(x       ��!�	���[`��A��3*i

A2S/kl�5�:

A2S/policy_network_losszA�

A2S/value_network_lossL`C

A2S/q_network_loss7cC�͖I,       ���E	�A�[`��A��3*

A2S/average_reward_1U�lAV��l,       ���E	f��[`��A��3*

A2S/average_reward_1yB��ڴ,       ���E	Z&�[`��A��3*

A2S/average_reward_1��Bo=,       ���E	�w \`��A��3*

A2S/average_reward_1���AQ�w�,       ���E	�\`��A��3*

A2S/average_reward_1w�AĐ��,       ���E	Y�\`��A��3*

A2S/average_reward_1��B�Y�,       ���E	��\`��A��3*

A2S/average_reward_1�C
zS,       ���E	AU�\`��A��3*

A2S/average_reward_1 _�Aǧ2�,       ���E	/>�\`��A��3*

A2S/average_reward_1���AȰ#?,       ���E	�\`��A��3*

A2S/average_reward_1��A���-,       ���E	N�\`��A��3*

A2S/average_reward_1֪C0
�,       ���E	v(]`��A��3*

A2S/average_reward_1��C(g\�,       ���E	[��]`��A��3*

A2S/average_reward_1�vD�!�0,       ���E	���]`��A��3*

A2S/average_reward_1w��A]�&,       ���E	�^`��A��3*

A2S/average_reward_1op�A+(),       ���E	B^`��A��3*

A2S/average_reward_1W��A6a,       ���E	��^`��A��3*

A2S/average_reward_1~��A��*I,       ���E	 �%^`��A��3*

A2S/average_reward_1�A${g�,       ���E	'�-^`��A��3*

A2S/average_reward_1n��A>Ō�,       ���E	���^`��A��3*

A2S/average_reward_1��`C{w|�,       ���E	���^`��A��3*

A2S/average_reward_1%ZB���,       ���E	#!_`��A��3*

A2S/average_reward_1�khC�"�,       ���E	q(_`��A��3*

A2S/average_reward_1�+�Au�d,       ���E	O�4_`��A��3*

A2S/average_reward_1�g�B�eFR,       ���E	�t_`��A��3*

A2S/average_reward_1/�Cn�o},       ���E	'|_`��A��3*

A2S/average_reward_1��A�YcJ,       ���E	�#``��A��3*

A2S/average_reward_1.��C�.�=,       ���E	�F$``��A��3*

A2S/average_reward_1�6B_t�,       ���E	sn�``��A��3*

A2S/average_reward_1��C}j��,       ���E	�C�``��A��3*

A2S/average_reward_1�$�@�x,       ���E	���``��A��3*

A2S/average_reward_134@B���,       ���E	v�aa`��A��3*

A2S/average_reward_1 ��C���H,       ���E	��ma`��A��3*

A2S/average_reward_1J��A�À,       ���E	�ya`��A��3*

A2S/average_reward_1�Bp>��,       ���E	c��a`��A��3*

A2S/average_reward_1�mFC��K$,       ���E	W��a`��A��3*

A2S/average_reward_1�sB5x�t,       ���E	/��a`��A��3*

A2S/average_reward_1�{B9@'�,       ���E	��a`��A��3*

A2S/average_reward_1��A�SK4,       ���E	���a`��A��3*

A2S/average_reward_1�[�A�U3�,       ���E	~�b`��A��3*

A2S/average_reward_1#B;l,       ���E	�bb`��A��3*

A2S/average_reward_1�%�A�A�b,       ���E	�AIb`��A��3*

A2S/average_reward_1��Cf]~�,       ���E	�^b`��A��3*

A2S/average_reward_1�WB=�,       ���E	gb`��A��3*

A2S/average_reward_1F��A���,       ���E	�زb`��A��3*

A2S/average_reward_1��!C�?�,       ���E	c�b`��A��3*

A2S/average_reward_1��A��d�,       ���E	TV�b`��A��3*

A2S/average_reward_12e�A��c�,       ���E	�`�b`��A��3*

A2S/average_reward_1�j�Atvt,       ���E	���b`��A��3*

A2S/average_reward_1�B
p�3,       ���E	�Dc`��A��3*

A2S/average_reward_1N��C�!B�,       ���E	��c`��A��3*

A2S/average_reward_1}�aC���&,       ���E	��c`��A��3*

A2S/average_reward_1lB��,       ���E	F*Nd`��A��3*

A2S/average_reward_10S�C�}Ș,       ���E	���d`��A��3*

A2S/average_reward_1 [�Bpt�,       ���E	 Ne`��A��4*

A2S/average_reward_1��C: [�,       ���E	�y|e`��A��4*

A2S/average_reward_1��B��?,       ���E	�]f`��A��4*

A2S/average_reward_1쏡C��/,       ���E	;�`g`��A�4*

A2S/average_reward_1��(DO3��,       ���E	��vg`��A��4*

A2S/average_reward_1�2B���,       ���E	w0�g`��A�4*

A2S/average_reward_1K�C�[�,       ���E	�r-h`��A�4*

A2S/average_reward_1�C �+�,       ���E	Z�i`��Aɜ4*

A2S/average_reward_1��,D�Mg�,       ���E	�ޙi`��A�4*

A2S/average_reward_1"��AO��C,       ���E	DB�i`��A��4*

A2S/average_reward_1/k�A��ˍ,       ���E	/M�i`��A��4*

A2S/average_reward_1n�A�U�h,       ���E	��i`��A̝4*

A2S/average_reward_1b��A]��,       ���E	��i`��A��4*

A2S/average_reward_1qB�5�\,       ���E	���j`��A��4*

A2S/average_reward_1j˩C�gH,       ���E	Sp k`��A��4*

A2S/average_reward_1�2�C:��,       ���E	��;k`��A��4*

A2S/average_reward_1���BL@&,       ���E	t� l`��A��4*

A2S/average_reward_1�ſCuWZv,       ���E	B[�l`��Aѭ4*

A2S/average_reward_1�܌C���-,       ���E	)�m`��A��4*

A2S/average_reward_1axC���L,       ���E	oJ*m`��A��4*

A2S/average_reward_1�!BB���,       ���E	��7m`��A��4*

A2S/average_reward_1I�BgN��,       ���E	��?m`��A��4*

A2S/average_reward_1�p�A	��,       ���E	9Bm`��Añ4*

A2S/average_reward_1peA)=e�,       ���E	�Om`��A�4*

A2S/average_reward_1oZ�A�+��,       ���E	��Um`��A��4*

A2S/average_reward_1c��AG���,       ���E	�j0n`��Aʶ4*

A2S/average_reward_1��Cf�ח,       ���E	��An`��A��4*

A2S/average_reward_1��*B��-t,       ���E	ZOn`��A��4*

A2S/average_reward_15M�A>T�6,       ���E	�V�n`��A��4*

A2S/average_reward_1�Cxۢ�,       ���E	�%�o`��Aȿ4*

A2S/average_reward_1�X�C%�!m,       ���E	���o`��A��4*

A2S/average_reward_1��BX�"�,       ���E	Na�o`��A��4*

A2S/average_reward_1�2=A�\w\,       ���E	�[/p`��A��4*

A2S/average_reward_13�C��I,       ���E	��Ap`��A��4*

A2S/average_reward_1*7A+��,       ���E	&�Lp`��A��4*

A2S/average_reward_1`�Ao:`,       ���E	T�Xp`��A��4*

A2S/average_reward_1VxV@P�,       ���E	�ap`��A��4*

A2S/average_reward_1�vfA�I��,       ���E	�X�p`��A��4*

A2S/average_reward_1#�C0��!,       ���E	���p`��A��4*

A2S/average_reward_1��B2��,       ���E	 �r`��A��4*

A2S/average_reward_1�r)D��#,       ���E	+z"r`��A��4*

A2S/average_reward_1�FB�	�(,       ���E	��Xs`��A��4*

A2S/average_reward_1GL.Dm�j,       ���E	!gfs`��A��4*

A2S/average_reward_1��AY^`�,       ���E	�x�s`��A��4*

A2S/average_reward_1-��B?���,       ���E	Fl�t`��A��4*

A2S/average_reward_1?+Duϓ�,       ���E	�G`v`��A��4*

A2S/average_reward_1�v.D���Y,       ���E	��gv`��A��4*

A2S/average_reward_1'��Al��S,       ���E	�Azv`��A��4*

A2S/average_reward_1�<�A[�զ,       ���E	��w`��A��4*

A2S/average_reward_1j�-D�cm�,       ���E	���w`��A��4*

A2S/average_reward_1�?�A�s��,       ���E	ӛ�w`��A��4*

A2S/average_reward_1��A�b#u,       ���E	Zg�w`��A��4*

A2S/average_reward_1�A�z� ,       ���E	���x`��A��4*

A2S/average_reward_1��CIX�,       ���E	�'�y`��A��4*

A2S/average_reward_1���C��>�,       ���E	i�;{`��A��5*

A2S/average_reward_1�--D�1=m,       ���E	X��|`��A��5*

A2S/average_reward_1[+DR��,       ���E	�#�}`��Aˏ5*

A2S/average_reward_1��D����,       ���E	�Ӹ}`��A��5*

A2S/average_reward_1�ukB��Ix,       ���E	6<�}`��A��5*

A2S/average_reward_1���A��f�,       ���E	/�~`��A�5*

A2S/average_reward_1�@C�OA^,       ���E	�X~`��A��5*

A2S/average_reward_1SA�Ƨz,       ���E	f�~`��A�5*

A2S/average_reward_1{��C�+�,       ���E	�ٺ`��A�5*

A2S/average_reward_1a��C�NXw,       ���E	�L�`��A��5*

A2S/average_reward_1�wNB��(�,       ���E	�)L�`��A��5*

A2S/average_reward_1)�.D�i�,       ���E	c;��`��Aĥ5*

A2S/average_reward_10�B��t�,       ���E	�-��`��A�5*

A2S/average_reward_1��B�t,       ���E	3oa�`��A��5*

A2S/average_reward_1��C���,       ���E	=�k�`��A۪5*

A2S/average_reward_1�"�A�+�,       ���E	v3��`��A٫5*

A2S/average_reward_1q
=B߳�%,       ���E	F'��`��A��5*

A2S/average_reward_1�"TBRS�,       ���E	%$��`��Aۭ5*

A2S/average_reward_1�CX�Pg,       ���E	��<�`��Aõ5*

A2S/average_reward_1�?*D9j,       ���E	eR��`��A��5*

A2S/average_reward_1f�-D,_�,       ���E	鸱�`��A޽5*

A2S/average_reward_1��)BET��,       ���E	�9Յ`��A��5*

A2S/average_reward_1��eBn%�,       ���E	�ޅ`��AҾ5*

A2S/average_reward_1�hA�VT,       ���E	��h�`��A��5*

A2S/average_reward_1EjsC��o,       ���E	����`��A��5*

A2S/average_reward_1�|"D�l�e,       ���E	�7�`��A��5*

A2S/average_reward_1��Cg���,       ���E	(��`��A��5*

A2S/average_reward_1�**Dd���,       ���E	��̉`��A��5*

A2S/average_reward_1��+C�*�t,       ���E	>�݉`��A��5*

A2S/average_reward_1!lA�wI',       ���E	8��`��A��5*

A2S/average_reward_1:)�A�>�,       ���E	:�`��A��5*

A2S/average_reward_1�SXAt� �,       ���E	?��`��A��5*

A2S/average_reward_1�.D�_5�,       ���E	�17�`��A��5*

A2S/average_reward_1wb/Cн�~,       ���E	���`��A��5*

A2S/average_reward_1~�&D�Nx�,       ���E	�`��A��5*

A2S/average_reward_1W�,D�6�Y,       ���E	�J�`��A��5*

A2S/average_reward_10��C���,       ���E	����`��A��5*

A2S/average_reward_1:�LB��b�,       ���E	�0�`��A��5*

A2S/average_reward_1�j�B�4��,       ���E	�6:�`��A��5*

A2S/average_reward_1M�e?���h,       ���E	ГT�`��A��5*

A2S/average_reward_1H\rBD$N,       ���E	J��`��A��5*

A2S/average_reward_1&s*D���,       ���E	{0U�`��A��6*

A2S/average_reward_1�~�C<[^5,       ���E	�'Q�`��A��6*

A2S/average_reward_1���C���t,       ���E	\�f�`��Aވ6*

A2S/average_reward_1	I�A
`��,       ���E	��r�`��A��6*

A2S/average_reward_1�㤿� =Z,       ���E	`Y�`��AŌ6*

A2S/average_reward_1	�C VmP,       ���E	*��`��A�6*

A2S/average_reward_1)�	B�1�,       ���E	�Z��`��A��6*

A2S/average_reward_1��CJ��g,       ���E	����`��A��6*

A2S/average_reward_1W�
B10x,       ���E	��`��A�6*

A2S/average_reward_1�	7C��T�,       ���E	���`��A�6*

A2S/average_reward_1�	�C��,       ���E	k���`��A�6*

A2S/average_reward_1�r�C�-�2,       ���E	��`��A��6*

A2S/average_reward_1��DB�Js,       ���E	��3�`��A��6*

A2S/average_reward_18DM��,       ���E	�A�`��A��6*

A2S/average_reward_1�a�AϹMF,       ���E	���`��Aī6*

A2S/average_reward_1�EC��N�,       ���E	ƹ��`��A��6*

A2S/average_reward_1k�B����,       ���E	��`��Aܳ6*

A2S/average_reward_1�F,D$^�,       ���E	O��`��A�6*

A2S/average_reward_1�8�A�B��,       ���E	3�4�`��A��6*

A2S/average_reward_1~��A�K��,       ���E	�@ǚ`��A��6*

A2S/average_reward_1iA�C|�+.,       ���E	|�Қ`��A׷6*

A2S/average_reward_1�B�5�,       ���E	�G�`��A��6*

A2S/average_reward_1��(B����,       ���E	-��`��A��6*

A2S/average_reward_12]B����,       ���E	�˛`��A߼6*

A2S/average_reward_1�d�C�e��,       ���E	(���`��A�6*

A2S/average_reward_1�+�B�!�n,       ���E	:+
�`��A��6*

A2S/average_reward_1��`A��1�,       ���E	���`��A��6*

A2S/average_reward_1u�A����,       ���E	H�ܜ`��A��6*

A2S/average_reward_1�;�C��K,       ���E	�y �`��A��6*

A2S/average_reward_1�V�B�ik�,       ���E	�e)�`��A��6*

A2S/average_reward_1�/�A�EU�,       ���E	g�=�`��A��6*

A2S/average_reward_1�.1B-W�,       ���E	x3��`��A��6*

A2S/average_reward_1��0C�;�,       ���E	U�`��A��6*

A2S/average_reward_1^I�C���,       ���E	�.c�`��A��6*

A2S/average_reward_1#�5B"`{,       ���E	�u��`��A��6*

A2S/average_reward_1�,DG�U,       ���E	�S��`��A��6*

A2S/average_reward_1@M�A�?��,       ���E	݇��`��A��6*

A2S/average_reward_1�.B-go�,       ���E	~1�`��A��6*

A2S/average_reward_1b�.D�E�,       ���E	gk1�`��A��6*

A2S/average_reward_1
��A"R�,       ���E	z��`��A��6*

A2S/average_reward_1H]C��i,       ���E	���`��A��6*

A2S/average_reward_1���Bĺl,       ���E	����`��A��6*

A2S/average_reward_1U��C��%A,       ���E	wH��`��A��6*

A2S/average_reward_1{F�C��[�,       ���E	�@�`��A��6*

A2S/average_reward_1��;Cd�U�,       ���E	j��`��A��6*

A2S/average_reward_1�zm@ڈx�,       ���E	
�1�`��A��6*

A2S/average_reward_1ZM{B�s��,       ���E	<�D�`��A��6*

A2S/average_reward_1�0�CB?C�,       ���E	�[Q�`��A��6*

A2S/average_reward_1� B쾰�,       ���E	#�\�`��A��6*

A2S/average_reward_1��A�|
,       ���E	��m�`��A��6*

A2S/average_reward_1� B1yY,       ���E	m�o�`��A��6*

A2S/average_reward_1��C�U�Z,       ���E	By�`��A��6*

A2S/average_reward_1���A�r�,       ���E	R��`��A��6*

A2S/average_reward_1X�&D���n,       ���E	����`��A��7*

A2S/average_reward_1ƿC����,       ���E	�z��`��A��7*

A2S/average_reward_1�ؗA<���,       ���E	7�è`��A΄7*

A2S/average_reward_1|�UAI�,       ���E	�b�`��A��7*

A2S/average_reward_1�-D�.�,       ���E	*8x�`��A��7*

A2S/average_reward_1���C|V�F,       ���E	�唫`��Aޗ7*

A2S/average_reward_1�*DL�_,       ���E	m^��`��A��7*

A2S/average_reward_1�1�A��,       ���E	`��A��7*

A2S/average_reward_1\��A���.,       ���E	V%�`��A��7*

A2S/average_reward_1'-D����,       ���E	�4�`��A��7*

A2S/average_reward_1%wD��F,       ���E	&+x�`��A��7*

A2S/average_reward_1XtC�p��,       ���E	p���`��AϨ7*

A2S/average_reward_13��A$ǵ?,       ���E	PƯ`��A��7*

A2S/average_reward_1u�(Dé�,       ���E	�MG�`��A��7*

A2S/average_reward_1N�tC�┮,       ���E	�O�`��A��7*

A2S/average_reward_1���A�B,       ���E	
�ð`��A��7*

A2S/average_reward_1�րCI�j},       ���E	�K��`��A��7*

A2S/average_reward_1c�Cdq��,       ���E	��`��A׼7*

A2S/average_reward_1�vaC���q,       ���E	�Z��`��A��7*

A2S/average_reward_1�x�A;ЮP,       ���E	�`	�`��A��7*

A2S/average_reward_1#�A����,       ���E	�98�`��A��7*

A2S/average_reward_1U>9B1d�,       ���E	�{S�`��Aξ7*

A2S/average_reward_1��A���?,       ���E	�4f�`��A��7*

A2S/average_reward_1�B���m,       ���E	�\�`��A��7*

A2S/average_reward_1, ]C�;y2,       ���E	���`��A��7*

A2S/average_reward_1O�A��,       ���E	�*�`��A��7*

A2S/average_reward_1"��C��,       ���E	�^�`��A��7*

A2S/average_reward_1W��A�u6�,       ���E	<��`��A��7*

A2S/average_reward_15�@Z] #,       ���E	�1�`��A��7*

A2S/average_reward_1gtB�ĕ�,       ���E	4�Ҵ`��A��7*

A2S/average_reward_1���Cfd ,       ���E	Ry?�`��A��7*

A2S/average_reward_1��+D�q�,       ���E	8�`��A��7*

A2S/average_reward_1b/C^e�,       ���E	��*�`��A��7*

A2S/average_reward_1��pC���,       ���E	���`��A��7*

A2S/average_reward_1!�C+�v�,       ���E	|�S�`��A��7*

A2S/average_reward_1�݇C�
�,       ���E	�L}�`��A��7*

A2S/average_reward_1yymB�@,       ���E	�s�`��A��7*

A2S/average_reward_1&�+D���,       ���E	us�`��A��7*

A2S/average_reward_14�CX�,       ���E	1A��`��A��7*

A2S/average_reward_1{�D���,       ���E	��t�`��A��7*

A2S/average_reward_1s��C���,       ���E	f��`��A��7*

A2S/average_reward_16�D�V|�,       ���E	z�Ӿ`��A��8*

A2S/average_reward_1�$*D씦:,       ���E	ҕ�`��A��8*

A2S/average_reward_1�Y�A�6\,       ���E	�%�`��A��8*

A2S/average_reward_1QC����,       ���E	0f�`��A��8*

A2S/average_reward_1���B7$�,       ���E	ơQ�`��A܍8*

A2S/average_reward_1,H�CH�P�,       ���E	��`�`��A��8*

A2S/average_reward_1=UB)u�t,       ���E	zUo�`��A��8*

A2S/average_reward_1�(B��A�,       ���E	9���`��A��8*

A2S/average_reward_1�J*D�
F,       ���E	9���`��Aܖ8*

A2S/average_reward_1�1B�x��,       ���E	*���`��A��8*

A2S/average_reward_1�X D��N�,       ���E	ڳ��`��A̡8*

A2S/average_reward_1�u�Cݒ �,       ���E	���`��A��8*

A2S/average_reward_1��&A��,       ���E	=� �`��A��8*

A2S/average_reward_1��wAd���,       ���E	P��`��AҢ8*

A2S/average_reward_1��B���,       ���E	,((�`��A��8*

A2S/average_reward_1�9�A$�d�,       ���E	�>3�`��A��8*

A2S/average_reward_1l¹A@��0,       ���E	$�<�`��A��8*

A2S/average_reward_1���A6^{,       ���E	k��`��Aʧ8*

A2S/average_reward_1g�C̍#,       ���E	+�&�`��A�8*

A2S/average_reward_1i�B��
�,       ���E	��t�`��Aۯ8*

A2S/average_reward_1��,DRCE,       ���E	�s�`��A�8*

A2S/average_reward_1,�D�f�,       ���E	sz��`��A�8*

A2S/average_reward_1��B��x       ��!�	�=�`��A�8*i

A2S/kl��<

A2S/policy_network_loss�Bο

A2S/value_network_loss��RC

A2S/q_network_loss��WC��,       ���E	�I��`��A��8*

A2S/average_reward_1�M�C�ic,       ���E	��`��A޻8*

A2S/average_reward_1��B��t,       ���E	���`��A��8*

A2S/average_reward_1(�D+( �,       ���E	���`��A��8*

A2S/average_reward_1���A��߄,       ���E	�-�`��A��8*

A2S/average_reward_1~lfB��,       ���E	�t��`��A��8*

A2S/average_reward_1w�C{�t,       ���E	r���`��A��8*

A2S/average_reward_1IOB���T,       ���E	���`��A��8*

A2S/average_reward_1
B��d,       ���E	���`��A��8*

A2S/average_reward_1���A��t,       ���E	w�`��A��8*

A2S/average_reward_1�p�C�,��,       ���E	3���`��A��8*

A2S/average_reward_1���C[�(,       ���E	>��`��A��8*

A2S/average_reward_1��Ar�On,       ���E	g��`��A��8*

A2S/average_reward_1�b�Ag�0�,       ���E	t�{�`��A��8*

A2S/average_reward_1��7D97��,       ���E	.���`��A��8*

A2S/average_reward_1->"BE�,       ���E	���`��A��8*

A2S/average_reward_1��Cib�,       ���E	2O��`��A��8*

A2S/average_reward_1��BP��,       ���E	< �`��A��8*

A2S/average_reward_1j��A:#�,       ���E	I�7�`��A��8*

A2S/average_reward_1$�7D
'�.,       ���E	��K�`��A��8*

A2S/average_reward_1��A� G,       ���E	7�R�`��A��8*

A2S/average_reward_1�	�A��6�,       ���E	��Z�`��A��8*

A2S/average_reward_1G�Ar͌�,       ���E	�`�`��A��8*

A2S/average_reward_1��A϶��,       ���E	�:l�`��A��8*

A2S/average_reward_1� �AS��4,       ���E	%X��`��A��8*

A2S/average_reward_1�C-�7�,       ���E	Uo	�`��A��8*

A2S/average_reward_1w4�A���[,       ���E	%��`��A��8*

A2S/average_reward_1e�BB���,       ���E	v��`��A��8*

A2S/average_reward_1pD��,       ���E	 r6�`��A��8*

A2S/average_reward_1�B\���,       ���E	�n?�`��A��8*

A2S/average_reward_1Ϸu?飯�,       ���E	�}��`��A��8*

A2S/average_reward_1�a�C�FZ,       ���E	w���`��A��8*

A2S/average_reward_14�A:P0,       ���E	�0�`��A��8*

A2S/average_reward_1"�9DfRbS,       ���E	�T`�`��A��8*

A2S/average_reward_1���B��H,       ���E	T�o�`��A��8*

A2S/average_reward_1��� ��,       ���E	�z�`��A��9*

A2S/average_reward_1ԘD>�,       ���E	��`��A��9*

A2S/average_reward_1���A��,       ���E	8���`��A��9*

A2S/average_reward_1ZX9D+Q�,       ���E	&�A�`��A�9*

A2S/average_reward_1kl6D�E�_,       ���E	,�I�`��A��9*

A2S/average_reward_1b5�A;d�F,       ���E	!Ҕ�`��A�9*

A2S/average_reward_1��6DI�,       ���E	����`��Aܙ9*

A2S/average_reward_1��:C.�YB,       ���E	�  �`��A��9*

A2S/average_reward_1���B֫�,       ���E	��`��A�9*

A2S/average_reward_1m3B�l�S,       ���E	��.�`��AǛ9*

A2S/average_reward_1�B[%�,       ���E	�v9�`��A�9*

A2S/average_reward_1�� B�8l,       ���E	jU��`��Aѣ9*

A2S/average_reward_13�7DM߻<,       ���E	6��`��A�9*

A2S/average_reward_1cc�A��R,       ���E	-���`��A��9*

A2S/average_reward_1�@�o�,       ���E	+L�`��A��9*

A2S/average_reward_1"�C�e?,       ���E	�j�`��A�9*

A2S/average_reward_1��8D���K,       ���E	�Mw�`��A��9*

A2S/average_reward_1�B0�e,       ���E	-���`��A�9*

A2S/average_reward_1 �xBG��,       ���E	�L��`��A��9*

A2S/average_reward_1}+5B�D �,       ���E	$��`��A��9*

A2S/average_reward_1�#�A�&�4x       ��!�	rG�`��A��9*i

A2S/kl�z:

A2S/policy_network_loss���

A2S/value_network_loss3�C

A2S/q_network_loss�s�C���	,       ���E	P�{�`��A��9*

A2S/average_reward_1]W�Bg��,       ���E	��`��A��9*

A2S/average_reward_1�yBdl��,       ���E	�g�`��A��9*

A2S/average_reward_1�"�C�ɤ,       ���E	c�U�`��A��9*

A2S/average_reward_1�T�B=���,       ���E	G]��`��A�9*

A2S/average_reward_1��qC���8,       ���E	��`��AӺ9*

A2S/average_reward_1,E:C�)<,       ���E	;�/�`��A»9*

A2S/average_reward_1_�B�,k�,       ���E	 ���`��A��9*

A2S/average_reward_1H�CA�Ӝ,       ���E	����`��Aʿ9*

A2S/average_reward_1�C�}^,       ���E	[�*�`��A��9*

A2S/average_reward_1HJC��,       ���E	,1��`��A��9*

A2S/average_reward_1Y�C��,       ���E	p��`��A��9*

A2S/average_reward_1%x�B�Ag�,       ���E	�(9�`��A��9*

A2S/average_reward_1-iC�� �,       ���E	��`�`��A��9*

A2S/average_reward_1�}Bu��,       ���E	D�o�`��A��9*

A2S/average_reward_1AxBCV�C,       ���E	2��`��A��9*

A2S/average_reward_1Q��BR��,       ���E	����`��A��9*

A2S/average_reward_1�Z�B�<��,       ���E	'e�`��A��9*

A2S/average_reward_1�L�C�c�[,       ���E	y��`��A��9*

A2S/average_reward_1�-C_oWb,       ���E	n�`�`��A��9*

A2S/average_reward_1��CݘPz,       ���E	2���`��A��9*

A2S/average_reward_1elC���,       ���E	Y��`��A��9*

A2S/average_reward_1a��B�$;N,       ���E	{M�`��A��9*

A2S/average_reward_1nU?�.��,       ���E	[M�`��A��9*

A2S/average_reward_1��C[�
�,       ���E		Y�`��A��9*

A2S/average_reward_1-�:B�_^,       ���E	X��`��A��9*

A2S/average_reward_1oߏClԥ�,       ���E	���`��A��9*

A2S/average_reward_1��B� u�,       ���E	��5�`��A��9*

A2S/average_reward_1	`�B���7,       ���E	����`��A��9*

A2S/average_reward_1aI9D�(,       ���E	#���`��A��9*

A2S/average_reward_162�A���,       ���E	 y0�`��A��9*

A2S/average_reward_1v�CSo
,       ���E	zV��`��A��9*

A2S/average_reward_1IMC�[�,       ���E	a>��`��A��9*

A2S/average_reward_1�4Bj��,       ���E	x)��`��A��9*

A2S/average_reward_1���A����,       ���E	�`��`��A��9*

A2S/average_reward_1��B��/�,       ���E	���`��A��9*

A2S/average_reward_1�GB�`�,       ���E	����`��A��9*

A2S/average_reward_1�N�A��<,       ���E	~���`��A��9*

A2S/average_reward_1�	�Ad��o,       ���E	�?��`��A��9*

A2S/average_reward_1)V�B'G�K,       ���E	���`��A��9*

A2S/average_reward_1��B��1,       ���E	��`��A��9*

A2S/average_reward_1��A�g,       ���E	����`��A��9*

A2S/average_reward_1���CM�,       ���E	C�`��A��9*

A2S/average_reward_1+�JC�Q3k,       ���E	��`��A��9*

A2S/average_reward_1޷BxCD,       ���E	ߣ��`��A��9*

A2S/average_reward_1&�mCa"0N,       ���E	O���`��A��9*

A2S/average_reward_1�1�BQ�wM,       ���E	�`% a��A��9*

A2S/average_reward_1|+8D(Ah,       ���E	-&o a��A��:*

A2S/average_reward_1�bC�,       ���E	�3� a��A��:*

A2S/average_reward_1w�:B
�0,       ���E	]�� a��AҀ:*

A2S/average_reward_1���A����,       ���E	M!� a��A��:*

A2S/average_reward_1p9�A�t�A,       ���E	�6� a��A��:*

A2S/average_reward_1�Bs�j�F,       ���E	!�� a��A��:*

A2S/average_reward_1�w�A��,       ���E	`�>a��AɄ:*

A2S/average_reward_1��C\�K�,       ���E	.�a��A��:*

A2S/average_reward_1k��C�p�,       ���E	��Ga��A�:*

A2S/average_reward_1^P-C���,       ���E	��Ra��A��:*

A2S/average_reward_1j�A���,       ���E	�fta��A�:*

A2S/average_reward_1��BM�,       ���E	P��a��A��:*

A2S/average_reward_1k�B �@o,       ���E	��a��A��:*

A2S/average_reward_1:�C�m-j,       ���E	�)a��A:*

A2S/average_reward_11WBu��*,       ���E	��a��A�:*

A2S/average_reward_1O��CAHXY,       ���E	 B)a��A͜:*

A2S/average_reward_1��&D>>��,       ���E	ԙ9a��A��:*

A2S/average_reward_1�݃AU05,       ���E	�Ba��A��:*

A2S/average_reward_1l�B�_�,       ���E	z�Ua��A�:*

A2S/average_reward_1�A�#,       ���E	�ٻa��Aϥ:*

A2S/average_reward_1�U+DV�	�,       ���E	�dba��Aݩ:*

A2S/average_reward_1-��C�ЛT,       ���E	�3na��A��:*

A2S/average_reward_1�B��U�,       ���E	��~a��A��:*

A2S/average_reward_1C��A6���,       ���E	tyya��A°:*

A2S/average_reward_1=�	D>�g,       ���E	|�	a��A��:*

A2S/average_reward_1Dx.D8��_,       ���E	�[
a��A��:*

A2S/average_reward_1���C�EB�,       ���E	_�y
a��A��:*

A2S/average_reward_1�gpC
�C,       ���E	ӕa��A��:*

A2S/average_reward_1��C�w�U,       ���E	h��a��A��:*

A2S/average_reward_1��CV0ƌ,       ���E	��a��A��:*

A2S/average_reward_1�̢A��+�,       ���E	���a��A��:*

A2S/average_reward_1�CJA�ݝ�,       ���E	g�Ha��A��:*

A2S/average_reward_1<o�C�L��,       ���E	%�Pa��A��:*

A2S/average_reward_1��A�*z>,       ���E	���a��A��:*

A2S/average_reward_1�ÿB�ަ,       ���E	ţ�a��A��:*

A2S/average_reward_1u�-D��e,       ���E	��a��A��:*

A2S/average_reward_1vyB�F�,       ���E	�-a��A��:*

A2S/average_reward_1x�,D �,       ���E	�S\a��A��:*

A2S/average_reward_17F�Bv��j,       ���E	�R�a��A��:*

A2S/average_reward_1��&CAEQ�,       ���E	�9�a��A��:*

A2S/average_reward_1��C�/�#,       ���E	�a��A��:*

A2S/average_reward_1��bC��,       ���E	Sa��A��:*

A2S/average_reward_1��C3��,       ���E	��[a��A��:*

A2S/average_reward_1=��Ah��,       ���E	P��a��A��:*

A2S/average_reward_1E.D�ǎ,       ���E	a�a��A��:*

A2S/average_reward_11D��/2,       ���E	m�a��A��:*

A2S/average_reward_1-aB0.��,       ���E	=2�a��A��:*

A2S/average_reward_1��-Bq��,       ���E	�Ra��A��:*

A2S/average_reward_1���Cv��T,       ���E	�^a��A��:*

A2S/average_reward_1	�Bg՟�,       ���E	� ;a��A��:*

A2S/average_reward_1m�D]J#�,       ���E	���a��A΁;*

A2S/average_reward_1i09C]�`,       ���E	xSa��A׆;*

A2S/average_reward_1DE�C�U,       ���E	
a��AՊ;*

A2S/average_reward_1D?�C>�%1,       ���E	<a��A��;*

A2S/average_reward_1u�BA�Wg�,       ���E	��fa��A�;*

A2S/average_reward_1�&D��S�,       ���E	�na��A��;*

A2S/average_reward_1[k�Ar~Uj,       ���E	�1wa��A��;*

A2S/average_reward_1���A�z4�,       ���E	;ƛa��Aș;*

A2S/average_reward_1��D�s(�,       ���E	���a��A��;*

A2S/average_reward_1M�)D���,       ���E	jp�a��A��;*

A2S/average_reward_1kF�C�ی�,       ���E	��a��A��;*

A2S/average_reward_1k�lBi�9,       ���E	��a��A��;*

A2S/average_reward_1���A�V�l,       ���E	ʃa��A��;*

A2S/average_reward_1J�A	J1,       ���E	(7a��A�;*

A2S/average_reward_1���A���x       ��!�	���'a��A�;*i

A2S/kl��8

A2S/policy_network_losszܿ

A2S/value_network_loss�SC

A2S/q_network_lossMVCF�M),       ���E	���'a��A��;*

A2S/average_reward_1��BV�U,       ���E	!�'a��A��;*

A2S/average_reward_1��A��C',       ���E	���'a��Aè;*

A2S/average_reward_1ܮ�AO �,       ���E	��(a��A��;*

A2S/average_reward_1�4C�Ԣ�,       ���E	��(a��A��;*

A2S/average_reward_1� �A�xr�,       ���E	�((a��Aɪ;*

A2S/average_reward_1T�B4>�,       ���E	��~(a��A��;*

A2S/average_reward_1<�2C~��W,       ���E	�]�(a��A��;*

A2S/average_reward_1�Bb�w,       ���E	J@�(a��A��;*

A2S/average_reward_1��A��,       ���E	nw�(a��A��;*

A2S/average_reward_1j2�B�{,       ���E	��)a��A��;*

A2S/average_reward_1}�5CW�E�,       ���E	c�o*a��A�;*

A2S/average_reward_14�<D$(�,       ���E	_K+a��A��;*

A2S/average_reward_1[ŰCs���,       ���E	n�+a��Až;*

A2S/average_reward_1�ͪCL�',       ���E	�(�+a��Aо;*

A2S/average_reward_1�A�kSd,       ���E		�+a��Aտ;*

A2S/average_reward_1���BV4-",       ���E	���+a��A��;*

A2S/average_reward_1��B�?�,       ���E	�#�+a��A��;*

A2S/average_reward_1p[	B�/yO,       ���E	I�d,a��A��;*

A2S/average_reward_1��C
$�;,       ���E	�~�,a��A��;*

A2S/average_reward_1˱&C�jD/,       ���E	��z-a��A��;*

A2S/average_reward_1J��C7v<�,       ���E	R�-a��A��;*

A2S/average_reward_1���BT�z*,       ���E	M�-a��A��;*

A2S/average_reward_1J��B��?,       ���E	bT�-a��A��;*

A2S/average_reward_1slB}��,       ���E	���-a��A��;*

A2S/average_reward_1۽�B�Ɲ�,       ���E	F.a��A��;*

A2S/average_reward_1d��B� �,       ���E	G.a��A��;*

A2S/average_reward_1;l
C�8\r,       ���E	�R.a��A��;*

A2S/average_reward_1�]BɊ�a,       ���E	kJ`.a��A��;*

A2S/average_reward_1���A����,       ���E	+'�.a��A��;*

A2S/average_reward_1%�]B	���,       ���E	�.a��A��;*

A2S/average_reward_1��6B�Z�z,       ���E	���/a��A��;*

A2S/average_reward_1^E1DkB
,       ���E	S��/a��A��;*

A2S/average_reward_1IkB^H�,       ���E	��/a��A��;*

A2S/average_reward_1�C�AFg/<,       ���E	��81a��A��;*

A2S/average_reward_1�7D&,       ���E	2y`1a��A��;*

A2S/average_reward_1���B]"�,       ���E	�}�1a��A��;*

A2S/average_reward_1�c�B5~\,       ���E	��1a��A��;*

A2S/average_reward_1���BL�lN,       ���E	�1a��A��;*

A2S/average_reward_1�o�B7d�,       ���E	w[2a��A��;*

A2S/average_reward_1���C�t�&,       ���E	�r2a��A��;*

A2S/average_reward_1��A�w�,       ���E	l�}2a��A��;*

A2S/average_reward_1�;B:��4,       ���E	kL�2a��A��;*

A2S/average_reward_1I=MB]�=p,       ���E	��2a��A��;*

A2S/average_reward_1n��B
���,       ���E	)3a��A��;*

A2S/average_reward_1�-HCV���,       ���E	��23a��A��;*

A2S/average_reward_1��B���D,       ���E	w�K3a��A��;*

A2S/average_reward_1	ޞB����,       ���E	K�v3a��A��;*

A2S/average_reward_1���B؉��,       ���E	�^�3a��A��;*

A2S/average_reward_1G��A0r�	,       ���E	��3a��A��;*

A2S/average_reward_1M1CQ#n,       ���E	]��3a��A��;*

A2S/average_reward_1��B�mY�,       ���E	��4a��A��;*

A2S/average_reward_1�n�B@�(,       ���E	�U(4a��A��;*

A2S/average_reward_1�_�A�~
E,       ���E	��@4a��A��;*

A2S/average_reward_1Q!qB��,       ���E	3�I4a��A��;*

A2S/average_reward_1���Ak�τ,       ���E	�n4a��A��;*

A2S/average_reward_1᜶BU!�,       ���E	���4a��A��;*

A2S/average_reward_1�IC�{t|,       ���E	��-6a��A��;*

A2S/average_reward_1A�*D�)$�,       ���E	�+:6a��A��;*

A2S/average_reward_1B�Z�],       ���E	sb�6a��A��;*

A2S/average_reward_1���Bq���,       ���E	�f�6a��A��;*

A2S/average_reward_1��A ��,       ���E	�2�6a��A��;*

A2S/average_reward_1M5A���,       ���E	{��6a��A��;*

A2S/average_reward_1g��A�y,       ���E	� 8a��A��<*

A2S/average_reward_1�(D��(,       ���E	o8a��Aڇ<*

A2S/average_reward_1�5�Ad���,       ���E	�bw9a��A<*

A2S/average_reward_1�5*D�ɗ�,       ���E	�0�9a��A��<*

A2S/average_reward_1=��A�X��,       ���E	I�9a��A��<*

A2S/average_reward_1�0�AWh�,       ���E	`�~:a��A��<*

A2S/average_reward_1r��C�}Y�,       ���E	�#;a��A��<*

A2S/average_reward_1�2�C��m,       ���E	*4�;a��A��<*

A2S/average_reward_1���C��*,       ���E	�#\<a��A��<*

A2S/average_reward_1E��C�G��,       ���E	Ըp<a��A��<*

A2S/average_reward_17B|[g�,       ���E	U#y<a��A��<*

A2S/average_reward_1���Ad���,       ���E	�6�<a��Aؤ<*

A2S/average_reward_1A9pC���,       ���E	]5>a��A��<*

A2S/average_reward_1;�+D�o��,       ���E	Q��?a��A��<*

A2S/average_reward_1L�+DA��,       ���E	x�?a��Aʹ<*

A2S/average_reward_13KB�C��,       ���E	��?a��A�<*

A2S/average_reward_1�=B�$��,       ���E	q8�?a��A��<*

A2S/average_reward_1c��AT�],       ���E	��?a��A��<*

A2S/average_reward_1,b�B�|,       ���E	,F�?a��A϶<*

A2S/average_reward_1'ԐA�=�,       ���E	Q�@a��A��<*

A2S/average_reward_1JhEBfG�",       ���E	aAa��A��<*

A2S/average_reward_1�I�C��]
,       ���E	t| Aa��Aļ<*

A2S/average_reward_1o�UB��O�,       ���E	�WxAa��Až<*

A2S/average_reward_1�<C�l,       ���E	�-�Aa��A��<*

A2S/average_reward_1@� A3k؟,       ���E	�
/Ba��A��<*

A2S/average_reward_1�ƯC��>,       ���E	ƹ�Ca��A��<*

A2S/average_reward_12�-D[K�b,       ���E	Aj�Ca��A��<*

A2S/average_reward_1wcCG�	�,       ���E	(�,Da��A��<*

A2S/average_reward_1�cC1��j,       ���E	��<Da��A��<*

A2S/average_reward_1Q��A� #�,       ���E	elLDa��A��<*

A2S/average_reward_1gglA,       ���E	�hDa��A��<*

A2S/average_reward_1�K	B�w�R,       ���E	��wDa��A��<*

A2S/average_reward_10KB��E,       ���E	W��Da��A��<*

A2S/average_reward_1L��Ao�+,       ���E	сEa��A��<*

A2S/average_reward_1F;\C1,��,       ���E	�[!Ea��A��<*

A2S/average_reward_1~�A��^,       ���E	��eFa��A��<*

A2S/average_reward_1sb,Dvg,       ���E	��Fa��A��<*

A2S/average_reward_1���B���,       ���E	1[�Ga��A��<*

A2S/average_reward_1�D0���,       ���E	��Ga��A��<*

A2S/average_reward_1�.aB}A=�,       ���E	�5UHa��A��<*

A2S/average_reward_1��C�l�",       ���E	��_Ha��A��<*

A2S/average_reward_1bh�AaYԴ,       ���E	Z0xHa��A��<*

A2S/average_reward_1�2�B@�J�,       ���E	���Ia��A��<*

A2S/average_reward_1u5.D����,       ���E	���Ia��A��<*

A2S/average_reward_1^G�Aii�,       ���E	�Ja��A��<*

A2S/average_reward_1T��C*�Ij,       ���E	��La��A��<*

A2S/average_reward_1��)D4���,       ���E	��sMa��A؃=*

A2S/average_reward_1d�,D��h�,       ���E	�Ma��A��=*

A2S/average_reward_1	(B��O�,       ���E	���Na��A�=*

A2S/average_reward_1*�,D>/�|,       ���E	\Oa��A��=*

A2S/average_reward_1
BYK�x       ��!�	qgZa��A��=*i

A2S/kl�\J;

A2S/policy_network_lossl'�

A2S/value_network_loss3�eC

A2S/q_network_loss�iCA�