       �K"	  �E�z�Abrain.Event:2C��ɕ;     ��_�	�1�E�z�A"��
s
A2S/observationsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
n
A2S/actionsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
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
X
A2S/mean_policy_oldPlaceholder*
dtype0*
_output_shapes
:*
shape:
Z
A2S/stddev_policy_oldPlaceholder*
shape:*
dtype0*
_output_shapes
:
n
A2S/returnsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
W
A2S/average_rewardPlaceholder*
dtype0*
_output_shapes
:*
shape:
�
VA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
seed2
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
�
PA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
�
5A2S/backup_policy_network/backup_policy_network/fc0/w
VariableV2*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
�
:A2S/backup_policy_network/backup_policy_network/fc0/w/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
�
GA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zerosConst*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
valueB*    *
dtype0
�
5A2S/backup_policy_network/backup_policy_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
	container *
shape:
�
<A2S/backup_policy_network/backup_policy_network/fc0/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/bGA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zeros*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
:A2S/backup_policy_network/backup_policy_network/fc0/b/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/b*
_output_shapes
:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b
�
 A2S/backup_policy_network/MatMulMatMulA2S/observations:A2S/backup_policy_network/backup_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_policy_network/addAdd A2S/backup_policy_network/MatMul:A2S/backup_policy_network/backup_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
:A2S/backup_policy_network/LayerNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
(A2S/backup_policy_network/LayerNorm/beta
VariableV2*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
/A2S/backup_policy_network/LayerNorm/beta/AssignAssign(A2S/backup_policy_network/LayerNorm/beta:A2S/backup_policy_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:
�
:A2S/backup_policy_network/LayerNorm/gamma/Initializer/onesConst*
_output_shapes
:*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
valueB*  �?*
dtype0
�
)A2S/backup_policy_network/LayerNorm/gamma
VariableV2*
shared_name *<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:
�
BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
8A2S/backup_policy_network/LayerNorm/moments/StopGradientStopGradient0A2S/backup_policy_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_policy_network/add8A2S/backup_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
FA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
x
3A2S/backup_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
1A2S/backup_policy_network/LayerNorm/batchnorm/addAdd4A2S/backup_policy_network/LayerNorm/moments/variance3A2S/backup_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
3A2S/backup_policy_network/LayerNorm/batchnorm/RsqrtRsqrt1A2S/backup_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
1A2S/backup_policy_network/LayerNorm/batchnorm/mulMul3A2S/backup_policy_network/LayerNorm/batchnorm/Rsqrt.A2S/backup_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_1MulA2S/backup_policy_network/add1A2S/backup_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2Mul0A2S/backup_policy_network/LayerNorm/moments/mean1A2S/backup_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
1A2S/backup_policy_network/LayerNorm/batchnorm/subSub-A2S/backup_policy_network/LayerNorm/beta/read3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
3A2S/backup_policy_network/LayerNorm/batchnorm/add_1Add3A2S/backup_policy_network/LayerNorm/batchnorm/mul_11A2S/backup_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
d
A2S/backup_policy_network/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *��?
�
A2S/backup_policy_network/mulMulA2S/backup_policy_network/mul/x3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
A2S/backup_policy_network/AbsAbs3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
f
!A2S/backup_policy_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_policy_network/mul_1Mul!A2S/backup_policy_network/mul_1/xA2S/backup_policy_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/backup_policy_network/add_1AddA2S/backup_policy_network/mulA2S/backup_policy_network/mul_1*'
_output_shapes
:���������*
T0
p
+A2S/backup_policy_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
VA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB"      
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *��̽
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *���=
�
^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
seed28
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
�
5A2S/backup_policy_network/backup_policy_network/out/w
VariableV2*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
<A2S/backup_policy_network/backup_policy_network/out/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/wPA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
:A2S/backup_policy_network/backup_policy_network/out/w/readIdentity5A2S/backup_policy_network/backup_policy_network/out/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
GA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
5A2S/backup_policy_network/backup_policy_network/out/b
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b
�
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b
�
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:*
T0
�
"A2S/backup_policy_network/MatMul_1MatMulA2S/backup_policy_network/add_1:A2S/backup_policy_network/backup_policy_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/backup_policy_network/add_2Add"A2S/backup_policy_network/MatMul_1:A2S/backup_policy_network/backup_policy_network/out/b/read*'
_output_shapes
:���������*
T0
�
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"      *
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
dtype0*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
seed2H
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:*
T0
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/fc0/w
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container 
�
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
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
:
�
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
8A2S/best_policy_network/LayerNorm/beta/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
&A2S/best_policy_network/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container 
�
-A2S/best_policy_network/LayerNorm/beta/AssignAssign&A2S/best_policy_network/LayerNorm/beta8A2S/best_policy_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
�
8A2S/best_policy_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*  �?
�
'A2S/best_policy_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:
�
.A2S/best_policy_network/LayerNorm/gamma/AssignAssign'A2S/best_policy_network/LayerNorm/gamma8A2S/best_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
,A2S/best_policy_network/LayerNorm/gamma/readIdentity'A2S/best_policy_network/LayerNorm/gamma*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0
�
@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
.A2S/best_policy_network/LayerNorm/moments/meanMeanA2S/best_policy_network/add@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
�
6A2S/best_policy_network/LayerNorm/moments/StopGradientStopGradient.A2S/best_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
DA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
v
1A2S/best_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
/A2S/best_policy_network/LayerNorm/batchnorm/addAdd2A2S/best_policy_network/LayerNorm/moments/variance1A2S/best_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtRsqrt/A2S/best_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
/A2S/best_policy_network/LayerNorm/batchnorm/mulMul1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
1A2S/best_policy_network/LayerNorm/batchnorm/mul_1MulA2S/best_policy_network/add/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
1A2S/best_policy_network/LayerNorm/batchnorm/mul_2Mul.A2S/best_policy_network/LayerNorm/moments/mean/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
/A2S/best_policy_network/LayerNorm/batchnorm/subSub+A2S/best_policy_network/LayerNorm/beta/read1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
1A2S/best_policy_network/LayerNorm/batchnorm/add_1Add1A2S/best_policy_network/LayerNorm/batchnorm/mul_1/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
b
A2S/best_policy_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/best_policy_network/mulMulA2S/best_policy_network/mul/x1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/best_policy_network/AbsAbs1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
d
A2S/best_policy_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/best_policy_network/mul_1MulA2S/best_policy_network/mul_1/xA2S/best_policy_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/best_policy_network/add_1AddA2S/best_policy_network/mulA2S/best_policy_network/mul_1*
T0*'
_output_shapes
:���������
n
)A2S/best_policy_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      *
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
dtype0*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2u
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes
: 
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
�
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/out/w
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:
�
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/add_16A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:���������
�
TA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB"      *
dtype0
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  �?
�
\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
3A2S/backup_value_network/backup_value_network/fc0/w
VariableV2*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
:A2S/backup_value_network/backup_value_network/fc0/w/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/wNA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
validate_shape(
�
8A2S/backup_value_network/backup_value_network/fc0/w/readIdentity3A2S/backup_value_network/backup_value_network/fc0/w*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
EA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zerosConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
3A2S/backup_value_network/backup_value_network/fc0/b
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b
�
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(
�
8A2S/backup_value_network/backup_value_network/fc0/b/readIdentity3A2S/backup_value_network/backup_value_network/fc0/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
_output_shapes
:
�
A2S/backup_value_network/MatMulMatMulA2S/observations8A2S/backup_value_network/backup_value_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_value_network/addAddA2S/backup_value_network/MatMul8A2S/backup_value_network/backup_value_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
9A2S/backup_value_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
valueB*    
�
'A2S/backup_value_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
	container *
shape:
�
.A2S/backup_value_network/LayerNorm/beta/AssignAssign'A2S/backup_value_network/LayerNorm/beta9A2S/backup_value_network/LayerNorm/beta/Initializer/zeros*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
,A2S/backup_value_network/LayerNorm/beta/readIdentity'A2S/backup_value_network/LayerNorm/beta*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
_output_shapes
:
�
9A2S/backup_value_network/LayerNorm/gamma/Initializer/onesConst*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
(A2S/backup_value_network/LayerNorm/gamma
VariableV2*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma
�
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:
�
AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
7A2S/backup_value_network/LayerNorm/moments/StopGradientStopGradient/A2S/backup_value_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_value_network/add7A2S/backup_value_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
EA2S/backup_value_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
w
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
0A2S/backup_value_network/LayerNorm/batchnorm/addAdd3A2S/backup_value_network/LayerNorm/moments/variance2A2S/backup_value_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/RsqrtRsqrt0A2S/backup_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
0A2S/backup_value_network/LayerNorm/batchnorm/mulMul2A2S/backup_value_network/LayerNorm/batchnorm/Rsqrt-A2S/backup_value_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
2A2S/backup_value_network/LayerNorm/batchnorm/mul_1MulA2S/backup_value_network/add0A2S/backup_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/mul_2Mul/A2S/backup_value_network/LayerNorm/moments/mean0A2S/backup_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
0A2S/backup_value_network/LayerNorm/batchnorm/subSub,A2S/backup_value_network/LayerNorm/beta/read2A2S/backup_value_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/add_1Add2A2S/backup_value_network/LayerNorm/batchnorm/mul_10A2S/backup_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
c
A2S/backup_value_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/backup_value_network/mulMulA2S/backup_value_network/mul/x2A2S/backup_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
A2S/backup_value_network/AbsAbs2A2S/backup_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
e
 A2S/backup_value_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_value_network/mul_1Mul A2S/backup_value_network/mul_1/xA2S/backup_value_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/backup_value_network/add_1AddA2S/backup_value_network/mulA2S/backup_value_network/mul_1*
T0*'
_output_shapes
:���������
o
*A2S/backup_value_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
TA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB"      
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2�
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
�
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:*
T0
�
3A2S/backup_value_network/backup_value_network/out/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
8A2S/backup_value_network/backup_value_network/out/w/readIdentity3A2S/backup_value_network/backup_value_network/out/w*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
�
EA2S/backup_value_network/backup_value_network/out/b/Initializer/zerosConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
3A2S/backup_value_network/backup_value_network/out/b
VariableV2*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
8A2S/backup_value_network/backup_value_network/out/b/readIdentity3A2S/backup_value_network/backup_value_network/out/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
_output_shapes
:
�
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_value_network/add_2Add!A2S/backup_value_network/MatMul_18A2S/backup_value_network/backup_value_network/out/b/read*
T0*'
_output_shapes
:���������
�
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"      *
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
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  �?*
dtype0
�
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes
: *
T0
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
�
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0
�
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container 
�
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
�
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
/A2S/best_value_network/best_value_network/fc0/b
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
�
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
�
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0
�
%A2S/best_value_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape:
�
,A2S/best_value_network/LayerNorm/beta/AssignAssign%A2S/best_value_network/LayerNorm/beta7A2S/best_value_network/LayerNorm/beta/Initializer/zeros*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
�
*A2S/best_value_network/LayerNorm/beta/readIdentity%A2S/best_value_network/LayerNorm/beta*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
7A2S/best_value_network/LayerNorm/gamma/Initializer/onesConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
&A2S/best_value_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container *
shape:
�
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
+A2S/best_value_network/LayerNorm/gamma/readIdentity&A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
�
?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
-A2S/best_value_network/LayerNorm/moments/meanMeanA2S/best_value_network/add?A2S/best_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
5A2S/best_value_network/LayerNorm/moments/StopGradientStopGradient-A2S/best_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
CA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
1A2S/best_value_network/LayerNorm/moments/varianceMean:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceCA2S/best_value_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
u
0A2S/best_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
.A2S/best_value_network/LayerNorm/batchnorm/addAdd1A2S/best_value_network/LayerNorm/moments/variance0A2S/best_value_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/RsqrtRsqrt.A2S/best_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
.A2S/best_value_network/LayerNorm/batchnorm/mulMul0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt+A2S/best_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_1MulA2S/best_value_network/add.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_2Mul-A2S/best_value_network/LayerNorm/moments/mean.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
.A2S/best_value_network/LayerNorm/batchnorm/subSub*A2S/best_value_network/LayerNorm/beta/read0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/add_1Add0A2S/best_value_network/LayerNorm/batchnorm/mul_1.A2S/best_value_network/LayerNorm/batchnorm/sub*'
_output_shapes
:���������*
T0
a
A2S/best_value_network/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *��?
�
A2S/best_value_network/mulMulA2S/best_value_network/mul/x0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
A2S/best_value_network/AbsAbs0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
c
A2S/best_value_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/best_value_network/mul_1MulA2S/best_value_network/mul_1/xA2S/best_value_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/best_value_network/add_1AddA2S/best_value_network/mulA2S/best_value_network/mul_1*'
_output_shapes
:���������*
T0
m
(A2S/best_value_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"      *
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
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *���=*
dtype0
�
XA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shape*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2�*
dtype0*
_output_shapes

:*

seed
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
/A2S/best_value_network/best_value_network/out/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
_output_shapes

:*
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:
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
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:���������
b
A2S/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
A2S/ReshapeReshapeA2S/backup_policy_network/add_2A2S/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
d
A2S/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
A2S/Reshape_1ReshapeA2S/best_policy_network/add_2A2S/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������
N
	A2S/ConstConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
P
A2S/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Y
A2S/Normal/locIdentityA2S/Reshape*
T0*'
_output_shapes
:���������
H
A2S/Normal/scaleIdentity	A2S/Const*
T0*
_output_shapes
: 
]
A2S/Normal_1/locIdentityA2S/Reshape_1*
T0*'
_output_shapes
:���������
L
A2S/Normal_1/scaleIdentityA2S/Const_1*
T0*
_output_shapes
: 
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
j
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal_1/scale*
T0*
_output_shapes
: 
j
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal/scale*
_output_shapes
: *
T0
�
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 
�
(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal_1/locA2S/Normal/loc*
T0*'
_output_shapes
:���������
�
-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*'
_output_shapes
:���������*
T0
�
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 
�
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*'
_output_shapes
:���������*
T0
�
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*
_output_shapes
: *
T0
~
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
T0*
_output_shapes
: 
�
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
_output_shapes
: *
T0
�
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
T0*
_output_shapes
: 
�
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*
T0*'
_output_shapes
:���������
\
A2S/Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
R
A2S/kl/tagsConst*
valueB BA2S/kl*
dtype0*
_output_shapes
: 
O
A2S/klScalarSummaryA2S/kl/tagsA2S/Mean*
_output_shapes
: *
T0
u
%A2S/Normal_2/batch_shape_tensor/ShapeShapeA2S/Normal_1/loc*
out_type0*
_output_shapes
:*
T0
j
'A2S/Normal_2/batch_shape_tensor/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
]
A2S/concat/values_0Const*
valueB:
*
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

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_2/batch_shape_tensor/BroadcastArgsA2S/concat/axis*

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
A2S/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*

seed*
T0*
dtype0*4
_output_shapes"
 :
������������������*
seed2�
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :
������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :
������������������
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*
T0*4
_output_shapes"
 :
������������������
h
A2S/addAddA2S/mulA2S/Normal_1/loc*
T0*4
_output_shapes"
 :
������������������
h
A2S/Reshape_2/shapeConst*!
valueB"����
      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*
Tshape0*+
_output_shapes
:���������
*
T0
S
A2S/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*

Tidx0*
T0*
N*'
_output_shapes
:���������
�
LA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/minConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
TA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
seed2�
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
FA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
�
+A2S/backup_q_network/backup_q_network/fc0/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/backup_q_network/fc0/b
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
2A2S/backup_q_network/backup_q_network/fc0/b/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/b=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b
�
0A2S/backup_q_network/backup_q_network/fc0/b/readIdentity+A2S/backup_q_network/backup_q_network/fc0/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
_output_shapes
:
�
A2S/backup_q_network/MatMulMatMulA2S/concat_10A2S/backup_q_network/backup_q_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_q_network/addAddA2S/backup_q_network/MatMul0A2S/backup_q_network/backup_q_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
5A2S/backup_q_network/LayerNorm/beta/Initializer/zerosConst*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
#A2S/backup_q_network/LayerNorm/beta
VariableV2*
_output_shapes
:*
shared_name *6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
	container *
shape:*
dtype0
�
*A2S/backup_q_network/LayerNorm/beta/AssignAssign#A2S/backup_q_network/LayerNorm/beta5A2S/backup_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
(A2S/backup_q_network/LayerNorm/beta/readIdentity#A2S/backup_q_network/LayerNorm/beta*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
_output_shapes
:
�
5A2S/backup_q_network/LayerNorm/gamma/Initializer/onesConst*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
$A2S/backup_q_network/LayerNorm/gamma
VariableV2*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
+A2S/backup_q_network/LayerNorm/gamma/AssignAssign$A2S/backup_q_network/LayerNorm/gamma5A2S/backup_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:
�
=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
3A2S/backup_q_network/LayerNorm/moments/StopGradientStopGradient+A2S/backup_q_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_q_network/add3A2S/backup_q_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
s
.A2S/backup_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
,A2S/backup_q_network/LayerNorm/batchnorm/addAdd/A2S/backup_q_network/LayerNorm/moments/variance.A2S/backup_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
.A2S/backup_q_network/LayerNorm/batchnorm/RsqrtRsqrt,A2S/backup_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
,A2S/backup_q_network/LayerNorm/batchnorm/mulMul.A2S/backup_q_network/LayerNorm/batchnorm/Rsqrt)A2S/backup_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
.A2S/backup_q_network/LayerNorm/batchnorm/mul_1MulA2S/backup_q_network/add,A2S/backup_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
.A2S/backup_q_network/LayerNorm/batchnorm/mul_2Mul+A2S/backup_q_network/LayerNorm/moments/mean,A2S/backup_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
,A2S/backup_q_network/LayerNorm/batchnorm/subSub(A2S/backup_q_network/LayerNorm/beta/read.A2S/backup_q_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
.A2S/backup_q_network/LayerNorm/batchnorm/add_1Add.A2S/backup_q_network/LayerNorm/batchnorm/mul_1,A2S/backup_q_network/LayerNorm/batchnorm/sub*'
_output_shapes
:���������*
T0
_
A2S/backup_q_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/backup_q_network/mulMulA2S/backup_q_network/mul/x.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
A2S/backup_q_network/AbsAbs.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
a
A2S/backup_q_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *���>
�
A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/backup_q_network/add_1AddA2S/backup_q_network/mulA2S/backup_q_network/mul_1*
T0*'
_output_shapes
:���������
k
&A2S/backup_q_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
LA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB"      *
dtype0
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/minConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
TA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
seed2�
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
�
FA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
�
+A2S/backup_q_network/backup_q_network/out/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
	container *
shape
:
�
2A2S/backup_q_network/backup_q_network/out/w/AssignAssign+A2S/backup_q_network/backup_q_network/out/wFA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
validate_shape(
�
0A2S/backup_q_network/backup_q_network/out/w/readIdentity+A2S/backup_q_network/backup_q_network/out/w*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:*
T0
�
=A2S/backup_q_network/backup_q_network/out/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/backup_q_network/out/b
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
2A2S/backup_q_network/backup_q_network/out/b/AssignAssign+A2S/backup_q_network/backup_q_network/out/b=A2S/backup_q_network/backup_q_network/out/b/Initializer/zeros*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
0A2S/backup_q_network/backup_q_network/out/b/readIdentity+A2S/backup_q_network/backup_q_network/out/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
_output_shapes
:
�
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/backup_q_network/add_2AddA2S/backup_q_network/MatMul_10A2S/backup_q_network/backup_q_network/out/b/read*
T0*'
_output_shapes
:���������
�
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"      *
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
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  �?*
dtype0
�
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2�
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
�
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:*
T0
�
'A2S/best_q_network/best_q_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:
�
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
�
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:*
T0
�
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    
�
'A2S/best_q_network/best_q_network/fc0/b
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:*
T0
�
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
3A2S/best_q_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    
�
!A2S/best_q_network/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container 
�
(A2S/best_q_network/LayerNorm/beta/AssignAssign!A2S/best_q_network/LayerNorm/beta3A2S/best_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
�
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
"A2S/best_q_network/LayerNorm/gamma
VariableV2*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
'A2S/best_q_network/LayerNorm/gamma/readIdentity"A2S/best_q_network/LayerNorm/gamma*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:
�
;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
�
1A2S/best_q_network/LayerNorm/moments/StopGradientStopGradient)A2S/best_q_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
6A2S/best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
q
,A2S/best_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
*A2S/best_q_network/LayerNorm/batchnorm/addAdd-A2S/best_q_network/LayerNorm/moments/variance,A2S/best_q_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/RsqrtRsqrt*A2S/best_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
*A2S/best_q_network/LayerNorm/batchnorm/mulMul,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt'A2S/best_q_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
,A2S/best_q_network/LayerNorm/batchnorm/mul_1MulA2S/best_q_network/add*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/mul_2Mul)A2S/best_q_network/LayerNorm/moments/mean*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
*A2S/best_q_network/LayerNorm/batchnorm/subSub&A2S/best_q_network/LayerNorm/beta/read,A2S/best_q_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/add_1Add,A2S/best_q_network/LayerNorm/batchnorm/mul_1*A2S/best_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
]
A2S/best_q_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/best_q_network/mulMulA2S/best_q_network/mul/x,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
}
A2S/best_q_network/AbsAbs,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
_
A2S/best_q_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/best_q_network/mul_1MulA2S/best_q_network/mul_1/xA2S/best_q_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/best_q_network/add_1AddA2S/best_q_network/mulA2S/best_q_network/mul_1*
T0*'
_output_shapes
:���������
i
$A2S/best_q_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"      *
dtype0
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

:*

seed*
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

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
�
'A2S/best_q_network/best_q_network/out/w
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container 
�
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
�
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
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
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
�
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/add_1,A2S/best_q_network/best_q_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:���������
}
%A2S/Normal_3/log_prob/standardize/subSubA2S/actionsA2S/Normal_1/loc*'
_output_shapes
:���������*
T0
�
)A2S/Normal_3/log_prob/standardize/truedivRealDiv%A2S/Normal_3/log_prob/standardize/subA2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
A2S/Normal_3/log_prob/SquareSquare)A2S/Normal_3/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
`
A2S/Normal_3/log_prob/mul/xConst*
valueB
 *   �*
dtype0*
_output_shapes
: 
�
A2S/Normal_3/log_prob/mulMulA2S/Normal_3/log_prob/mul/xA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:���������
U
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
T0*
_output_shapes
: 
`
A2S/Normal_3/log_prob/add/xConst*
valueB
 *�?k?*
dtype0*
_output_shapes
: 
y
A2S/Normal_3/log_prob/addAddA2S/Normal_3/log_prob/add/xA2S/Normal_3/log_prob/Log*
T0*
_output_shapes
: 
�
A2S/Normal_3/log_prob/subSubA2S/Normal_3/log_prob/mulA2S/Normal_3/log_prob/add*'
_output_shapes
:���������*
T0
[
A2S/NegNegA2S/Normal_3/log_prob/sub*
T0*'
_output_shapes
:���������
[
	A2S/mul_1MulA2S/NegA2S/advantages*'
_output_shapes
:���������*
T0
\
A2S/Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
m

A2S/Mean_1MeanA2S/advantagesA2S/Const_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
A2S/average_advantage/tagsConst*&
valueB BA2S/average_advantage*
dtype0*
_output_shapes
: 
o
A2S/average_advantageScalarSummaryA2S/average_advantage/tags
A2S/Mean_1*
T0*
_output_shapes
: 
\
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
h

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
t
A2S/policy_network_loss/tagsConst*(
valueB BA2S/policy_network_loss*
dtype0*
_output_shapes
: 
s
A2S/policy_network_lossScalarSummaryA2S/policy_network_loss/tags
A2S/Mean_2*
T0*
_output_shapes
: 
�
A2S/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add_2A2S/returns*'
_output_shapes
:���������*
T0
\
A2S/Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
t

A2S/Mean_3MeanA2S/SquaredDifferenceA2S/Const_5*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
r
A2S/value_network_loss/tagsConst*'
valueB BA2S/value_network_loss*
dtype0*
_output_shapes
: 
q
A2S/value_network_lossScalarSummaryA2S/value_network_loss/tags
A2S/Mean_3*
T0*
_output_shapes
: 
�
A2S/SquaredDifference_1SquaredDifferenceA2S/best_q_network/add_2A2S/returns*
T0*'
_output_shapes
:���������
\
A2S/Const_6Const*
dtype0*
_output_shapes
:*
valueB"       
v

A2S/Mean_4MeanA2S/SquaredDifference_1A2S/Const_6*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
j
A2S/q_network_loss/tagsConst*#
valueB BA2S/q_network_loss*
dtype0*
_output_shapes
: 
i
A2S/q_network_lossScalarSummaryA2S/q_network_loss/tags
A2S/Mean_4*
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
+A2S/gradients/A2S/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
%A2S/gradients/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
#A2S/gradients/A2S/Mean_2_grad/ShapeShape	A2S/mul_1*
out_type0*
_output_shapes
:*
T0
�
"A2S/gradients/A2S/Mean_2_grad/TileTile%A2S/gradients/A2S/Mean_2_grad/Reshape#A2S/gradients/A2S/Mean_2_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
n
%A2S/gradients/A2S/Mean_2_grad/Shape_1Shape	A2S/mul_1*
_output_shapes
:*
T0*
out_type0
h
%A2S/gradients/A2S/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
m
#A2S/gradients/A2S/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"A2S/gradients/A2S/Mean_2_grad/ProdProd%A2S/gradients/A2S/Mean_2_grad/Shape_1#A2S/gradients/A2S/Mean_2_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
o
%A2S/gradients/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
i
'A2S/gradients/A2S/Mean_2_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
%A2S/gradients/A2S/Mean_2_grad/MaximumMaximum$A2S/gradients/A2S/Mean_2_grad/Prod_1'A2S/gradients/A2S/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
�
&A2S/gradients/A2S/Mean_2_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_2_grad/Prod%A2S/gradients/A2S/Mean_2_grad/Maximum*
_output_shapes
: *
T0
�
"A2S/gradients/A2S/Mean_2_grad/CastCast&A2S/gradients/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
%A2S/gradients/A2S/Mean_2_grad/truedivRealDiv"A2S/gradients/A2S/Mean_2_grad/Tile"A2S/gradients/A2S/Mean_2_grad/Cast*
T0*'
_output_shapes
:���������
i
"A2S/gradients/A2S/mul_1_grad/ShapeShapeA2S/Neg*
out_type0*
_output_shapes
:*
T0
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
 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_2_grad/truedivA2S/advantages*'
_output_shapes
:���������*
T0
�
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
:���������*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape
�
7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1
�
A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ShapeShapeA2S/Normal_3/log_prob/mul*
T0*
out_type0*
_output_shapes
:
w
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
BA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1*
_output_shapes
:*
T0
�
6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
=A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape*'
_output_shapes
:���������
�
GA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1*
_output_shapes
: *
T0
u
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1ShapeA2S/Normal_3/log_prob/Square*
out_type0*
_output_shapes
:*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:���������
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1MulA2S/Normal_3/log_prob/mul/xEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
=A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape*
_output_shapes
: 
�
GA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
3A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/x)A2S/Normal_3/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul*
T0*'
_output_shapes
:���������
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_3/log_prob/standardize/sub*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
RA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1A2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
MA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1
�
UA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:���������
�
WA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
_output_shapes
:*
T0*
out_type0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal_1/loc*
T0*
out_type0*
_output_shapes
:
�
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
_output_shapes
:*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
IA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1
�
QA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape
�
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:���������
�
&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
out_type0*
_output_shapes
:*
T0
�
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/ShapeShape A2S/best_policy_network/MatMul_1*
out_type0*
_output_shapes
:*
T0
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
AA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
KA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1
�
:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
DA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul=^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
�
LA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulE^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul
�
NA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1E^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*O
_classE
CAloc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/ShapeShapeA2S/best_policy_network/mul*
T0*
out_type0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1ShapeA2S/best_policy_network/mul_1*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
AA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape
�
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
w
4A2S/gradients/A2S/best_policy_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/mul_grad/Shape6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2A2S/gradients/A2S/best_policy_network/mul_grad/mulMulIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1MulA2S/best_policy_network/mul/xIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_1Sum4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1FA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_16A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
?A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape9^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1
�
GA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/mul_grad/Reshape@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape*
_output_shapes
: 
�
IA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*'
_output_shapes
:���������
y
6A2S/gradients/A2S/best_policy_network/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1ShapeA2S/best_policy_network/Abs*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulMulKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1A2S/best_policy_network/Abs*'
_output_shapes
:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
AA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
3A2S/gradients/A2S/best_policy_network/Abs_grad/SignSign1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/best_policy_network/Abs_grad/mulMulKA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_13A2S/gradients/A2S/best_policy_network/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*
N*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_policy_network/add]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
_output_shapes
:*
T0*
out_type0
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegNegHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape.A2S/best_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul.A2S/best_policy_network/LayerNorm/moments/mean]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:���������
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients/AddN_1AddN_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtA2S/gradients/AddN_1*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape2A2S/best_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumSumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeShape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addAddDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modFloorModIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeRangeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0
�
PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/FillFillMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumSA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:���������*
T0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeReshape[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencySA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileTileMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeNA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape2A2S/best_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1MaximumLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/CastCastPA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truedivRealDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Cast*'
_output_shapes
:���������*
T0
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
�
dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulUA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradientN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpW^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeS^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
gA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*i
_class_
][loc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape
�
iA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*e
_class[
YWloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addAdd@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modFloorModEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeRangeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/delta*

Tidx0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/FillFillIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
�
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/MaximumMaximumOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeReshape]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileTileIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape.A2S/best_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumHA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/CastCastLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truedivRealDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:���������*
T0
�
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/add_grad/ShapeShapeA2S/best_policy_network/MatMul*
out_type0*
_output_shapes
:*
T0
�
6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/add_grad/Sum_16A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
?A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/add_grad/Reshape9^A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1
�
GA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/add_grad/Reshape@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape*'
_output_shapes
:���������
�
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
_output_shapes
:*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1
�
8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
BA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul;^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1
�
JA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulC^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
LA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1C^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:
�
A2S/beta1_power/initial_valueConst*
valueB
 *fff?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power/readIdentityA2S/beta1_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/beta2_power/initial_valueConst*
valueB
 *w�?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta2_power
VariableV2*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power/readIdentityA2S/beta2_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
LA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container 
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/w/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
LA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/b/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container 
�
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
AA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
/A2S/A2S/best_policy_network/LayerNorm/beta/Adam
VariableV2*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
4A2S/A2S/best_policy_network/LayerNorm/beta/Adam/readIdentity/A2S/A2S/best_policy_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1
VariableV2*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/AssignAssign1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/readIdentity1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
�
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container 
�
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
�
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
�
DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
�
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/readIdentity2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
�
LA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/w/AdamLA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(
�
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
LA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:
�
CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
S
A2S/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
S
A2S/Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
U
A2S/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/w:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/b:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonIA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
@A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam&A2S/best_policy_network/LayerNorm/beta/A2S/A2S/best_policy_network/LayerNorm/beta/Adam1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
�
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/w:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/b:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonKA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
_output_shapes
: *
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
�
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/AdamNoOpL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
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
-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
'A2S/gradients_1/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
z
%A2S/gradients_1/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference*
out_type0*
_output_shapes
:*
T0
�
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
|
'A2S/gradients_1/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_1/A2S/Mean_3_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%A2S/gradients_1/A2S/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_3_grad/ProdProd'A2S/gradients_1/A2S/Mean_3_grad/Shape_1%A2S/gradients_1/A2S/Mean_3_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
q
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
k
)A2S/gradients_1/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_1/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_3_grad/Prod_1)A2S/gradients_1/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
�
(A2S/gradients_1/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_3_grad/Prod'A2S/gradients_1/A2S/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
�
$A2S/gradients_1/A2S/Mean_3_grad/CastCast(A2S/gradients_1/A2S/Mean_3_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add_2*
out_type0*
_output_shapes
:*
T0
}
2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1ShapeA2S/returns*
T0*
out_type0*
_output_shapes
:
�
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/best_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( 
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
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
BA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
LA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulMatMulJA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1MatMulA2S/best_value_network/add_1JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
EA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_depsNoOp<^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul>^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1
�
MA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIdentity;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulF^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
OA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1Identity=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1F^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1
�
7A2S/gradients_1/A2S/best_value_network/add_1_grad/ShapeShapeA2S/best_value_network/mul*
_output_shapes
:*
T0*
out_type0
�
9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1ShapeA2S/best_value_network/mul_1*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients_1/A2S/best_value_network/add_1_grad/SumSumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_1SumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
BA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape*'
_output_shapes
:���������
�
LA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1*'
_output_shapes
:���������
x
5A2S/gradients_1/A2S/best_value_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
EA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/mul_grad/mulMulJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/mul_grad/SumSum3A2S/gradients_1/A2S/best_value_network/mul_grad/mulEA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/mul_grad/Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1MulA2S/best_value_network/mul/xJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
@A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1
�
HA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1
z
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1ShapeA2S/best_value_network/Abs*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulMulLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1A2S/best_value_network/Abs*
T0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1MulA2S/best_value_network/mul_1/xLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
BA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape
�
LA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
4A2S/gradients_1/A2S/best_value_network/Abs_grad/SignSign0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_1/A2S/best_value_network/Abs_grad/mulMulLA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_14A2S/gradients_1/A2S/best_value_network/Abs_grad/Sign*'
_output_shapes
:���������*
T0
�
A2S/gradients_1/AddNAddNJA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_13A2S/gradients_1/A2S/best_value_network/Abs_grad/mul*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*
N*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
_output_shapes
:*
T0*
out_type0
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegNegIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-A2S/best_value_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-A2S/best_value_network/LayerNorm/moments/mean^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:���������
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients_1/AddN_1AddN`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0A2S/best_value_network/LayerNorm/batchnorm/RsqrtA2S/gradients_1/AddN_1*
T0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeShape1A2S/best_value_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addAddCA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modFloorModJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeRangeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*
_output_shapes
:
�
QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/MaximumMaximumTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:���������*
T0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordivFloorDivLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1MaximumMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0
�
QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/CastCastQA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truedivRealDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5A2S/best_value_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
�
eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarConstO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulMulVA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradientO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegYA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpX^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeT^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
hA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshapea^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*j
_class`
^\loc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape
�
jA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitySA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Nega^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*f
_class\
ZXloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
�
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addAdd?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modFloorModFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeRangeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/FillFillJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/MaximumMaximumPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
T0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeReshape^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3Shape-A2S/best_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1ProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumIA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/CastCastMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truedivRealDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������
�
A2S/gradients_1/AddN_2AddN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyhA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truediv*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/add_grad/ShapeShapeA2S/best_value_network/MatMul*
_output_shapes
:*
T0*
out_type0
�
7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_17A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
@A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1
�
HA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape*'
_output_shapes
:���������
�
JA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
CA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul<^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1
�
KA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulD^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
MA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1D^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1
�
A2S/beta1_power_1/initial_valueConst*
valueB
 *fff?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_1
VariableV2*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/beta2_power_1/initial_valueConst*
valueB
 *w�?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta2_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container 
�
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
�
JA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
=A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0
�
LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    
�
:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container 
�
AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
�
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0
�
8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
�
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/b/AdamJA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
�
=A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
�
LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
�
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
�
@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
.A2S/A2S/best_value_network/LayerNorm/beta/Adam
VariableV2*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    
�
0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/AssignAssign0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
5A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/readIdentity0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
AA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
/A2S/A2S/best_value_network/LayerNorm/gamma/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container *
shape:
�
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/AssignAssign/A2S/A2S/best_value_network/LayerNorm/gamma/AdamAA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
�
4A2S/A2S/best_value_network/LayerNorm/gamma/Adam/readIdentity/A2S/A2S/best_value_network/LayerNorm/gamma/Adam*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:
�
CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1
VariableV2*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:
�
JA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
8A2S/A2S/best_value_network/best_value_network/out/w/Adam
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
=A2S/A2S/best_value_network/best_value_network/out/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/w/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
�
LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zerosConst*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    *
dtype0
�
:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
�
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    
�
8A2S/A2S/best_value_network/best_value_network/out/b/Adam
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
?A2S/A2S/best_value_network/best_value_network/out/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/b/AdamJA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
use_nesterov( *
_output_shapes

:
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/b8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonJA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
�
BA2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&A2S/best_value_network/LayerNorm/gamma/A2S/A2S/best_value_network/LayerNorm/gamma/Adam1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/w8A2S/A2S/best_value_network/best_value_network/out/w/Adam:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�

A2S/Adam_1NoOpL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
X
A2S/gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
'A2S/gradients_2/A2S/Mean_4_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
|
%A2S/gradients_2/A2S/Mean_4_grad/ShapeShapeA2S/SquaredDifference_1*
out_type0*
_output_shapes
:*
T0
�
$A2S/gradients_2/A2S/Mean_4_grad/TileTile'A2S/gradients_2/A2S/Mean_4_grad/Reshape%A2S/gradients_2/A2S/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
~
'A2S/gradients_2/A2S/Mean_4_grad/Shape_1ShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_2/A2S/Mean_4_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
o
%A2S/gradients_2/A2S/Mean_4_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_2/A2S/Mean_4_grad/ProdProd'A2S/gradients_2/A2S/Mean_4_grad/Shape_1%A2S/gradients_2/A2S/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
q
'A2S/gradients_2/A2S/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
k
)A2S/gradients_2/A2S/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_2/A2S/Mean_4_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_4_grad/Prod_1)A2S/gradients_2/A2S/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 
�
(A2S/gradients_2/A2S/Mean_4_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_4_grad/Prod'A2S/gradients_2/A2S/Mean_4_grad/Maximum*
T0*
_output_shapes
: 
�
$A2S/gradients_2/A2S/Mean_4_grad/CastCast(A2S/gradients_2/A2S/Mean_4_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
'A2S/gradients_2/A2S/Mean_4_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_4_grad/Tile$A2S/gradients_2/A2S/Mean_4_grad/Cast*'
_output_shapes
:���������*
T0
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
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
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_4_grad/truediv*'
_output_shapes
:���������*
T0
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/best_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
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
:*

Tidx0*
	keep_dims( 
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
:*

Tidx0*
	keep_dims( 
�
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
3A2S/gradients_2/A2S/best_q_network/add_2_grad/ShapeShapeA2S/best_q_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
CA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
>A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape
�
HA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1*
_output_shapes
:
�
7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1MatMulA2S/best_q_network/add_1FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
AA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_depsNoOp8^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul:^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1
�
IA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyIdentity7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulB^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
KA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1Identity9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1B^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*L
_classB
@>loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1
�
3A2S/gradients_2/A2S/best_q_network/add_1_grad/ShapeShapeA2S/best_q_network/mul*
out_type0*
_output_shapes
:*
T0
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1ShapeA2S/best_q_network/mul_1*
T0*
out_type0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_1SumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
HA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1*'
_output_shapes
:���������
t
1A2S/gradients_2/A2S/best_q_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
AA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
/A2S/gradients_2/A2S/best_q_network/mul_grad/mulMulFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
/A2S/gradients_2/A2S/best_q_network/mul_grad/SumSum/A2S/gradients_2/A2S/best_q_network/mul_grad/mulAA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
3A2S/gradients_2/A2S/best_q_network/mul_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/mul_grad/Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1MulA2S/best_q_network/mul/xFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
<A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape*
_output_shapes
: 
�
FA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*'
_output_shapes
:���������
v
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1ShapeA2S/best_q_network/Abs*
_output_shapes
:*
T0*
out_type0
�
CA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulMulHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1A2S/best_q_network/Abs*
T0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
>A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape
�
HA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/best_q_network/Abs_grad/SignSign,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
/A2S/gradients_2/A2S/best_q_network/Abs_grad/mulMulHA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_10A2S/gradients_2/A2S/best_q_network/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
A2S/gradients_2/AddNAddNFA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1/A2S/gradients_2/A2S/best_q_network/Abs_grad/mul*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*
N*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
out_type0*
_output_shapes
:*
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul)A2S/best_q_network/LayerNorm/moments/meanZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:���������
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients_2/AddN_1AddN\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad,A2S/best_q_network/LayerNorm/batchnorm/RsqrtXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1SumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeShape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addAdd?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modFloorModFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*
_output_shapes
:
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/FillFillJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordivFloorDivHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeReshapeXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2Shape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3Shape-A2S/best_q_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*
dtype0
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/CastCastMA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truedivRealDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Cast*'
_output_shapes
:���������*
T0
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
�
aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeSA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulMulRA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradientK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1cA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/NegNegUA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpT^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeP^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
dA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*f
_class\
ZXloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape
�
fA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*b
_classX
VTloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addAdd;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modFloorModBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeRangeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/FillFillFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill*#
_output_shapes
:���������*
T0*
N
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/MaximumMaximumLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordivFloorDivDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum*
T0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
out_type0*
_output_shapes
:*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1ProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/CastCastIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������
�
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/add_grad/ShapeShapeA2S/best_q_network/MatMul*
T0*
out_type0*
_output_shapes
:
}
3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/add_grad/Shape3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/A2S/gradients_2/A2S/best_q_network/add_grad/SumSumA2S/gradients_2/AddN_2AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
<A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/add_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape*'
_output_shapes
:���������*
T0
�
FA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
?A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul8^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
�
GA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
IA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
�
A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_2
VariableV2*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(
�
A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/beta2_power_2/initial_valueConst*
valueB
 *w�?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta2_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape: 
�
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
BA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:
�
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/w/AdamBA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
�
5A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
�
DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(
�
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
BA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container 
�
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/b/AdamBA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
�
5A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
�
DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    
�
2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
�
<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
*A2S/A2S/best_q_network/LayerNorm/beta/Adam
VariableV2*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
1A2S/A2S/best_q_network/LayerNorm/beta/Adam/AssignAssign*A2S/A2S/best_q_network/LayerNorm/beta/Adam<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
/A2S/A2S/best_q_network/LayerNorm/beta/Adam/readIdentity*A2S/A2S/best_q_network/LayerNorm/beta/Adam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
�
>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1
VariableV2*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
3A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/AssignAssign,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
1A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/readIdentity,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
�
=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
+A2S/A2S/best_q_network/LayerNorm/gamma/Adam
VariableV2*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/AssignAssign+A2S/A2S/best_q_network/LayerNorm/gamma/Adam=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
0A2S/A2S/best_q_network/LayerNorm/gamma/Adam/readIdentity+A2S/A2S/best_q_network/LayerNorm/gamma/Adam*
_output_shapes
:*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1
VariableV2*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/readIdentity-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:
�
BA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
0A2S/A2S/best_q_network/best_q_network/out/w/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container 
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/w/AdamBA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
�
5A2S/A2S/best_q_network/best_q_network/out/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/w/Adam*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zerosConst*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0
�
2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container 
�
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
�
BA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0
�
0A2S/A2S/best_q_network/best_q_network/out/b/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
�
7A2S/A2S/best_q_network/best_q_network/out/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/b/AdamBA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
5A2S/A2S/best_q_network/best_q_network/out/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/b/Adam*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
�
DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1
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
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(
�
7A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
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
A2S/Adam_2/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( *
_output_shapes

:
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/b0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonFA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
use_nesterov( *
_output_shapes
:
�
=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/w0A2S/A2S/best_q_network/best_q_network/out/w/Adam2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
use_nesterov( *
_output_shapes

:
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/b0A2S/A2S/best_q_network/best_q_network/out/b/Adam2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonHA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
_output_shapes
: *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( 
�

A2S/Adam_2NoOpD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1

A2S/group_depsNoOp

A2S/group_deps_1NoOp
�
A2S/Merge/MergeSummaryMergeSummaryA2S/klA2S/average_advantageA2S/policy_network_lossA2S/value_network_lossA2S/q_network_loss*
_output_shapes
: *
N
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
T0"��Ϳ�     (+�E	�خE�z�AJ��
��
+
Abs
x"T
y"T"
Ttype:	
2	
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
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
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
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
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
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
-
Rsqrt
x"T
y"T"
Ttype:	
2
9
	RsqrtGrad
x"T
y"T
z"T"
Ttype:	
2
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
.
Sign
x"T
y"T"
Ttype:
	2	
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
2
StopGradient

input"T
output"T"	
Ttype
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
shared_namestring �*1.3.02v1.3.0-rc2-20-g0787eee��
s
A2S/observationsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
q
A2S/advantagesPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
V
A2S/learning_ratePlaceholder*
shape:*
dtype0*
_output_shapes
:
X
A2S/mean_policy_oldPlaceholder*
dtype0*
_output_shapes
:*
shape:
Z
A2S/stddev_policy_oldPlaceholder*
shape:*
dtype0*
_output_shapes
:
n
A2S/returnsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
W
A2S/average_rewardPlaceholder*
_output_shapes
:*
shape:*
dtype0
�
VA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  �?*
dtype0
�
^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
seed2*
dtype0*
_output_shapes

:*

seed
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
�
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
�
PA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
�
5A2S/backup_policy_network/backup_policy_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
	container *
shape
:
�
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
:A2S/backup_policy_network/backup_policy_network/fc0/w/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
�
GA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
5A2S/backup_policy_network/backup_policy_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
	container *
shape:
�
<A2S/backup_policy_network/backup_policy_network/fc0/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/bGA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
validate_shape(*
_output_shapes
:
�
:A2S/backup_policy_network/backup_policy_network/fc0/b/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
_output_shapes
:
�
 A2S/backup_policy_network/MatMulMatMulA2S/observations:A2S/backup_policy_network/backup_policy_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/backup_policy_network/addAdd A2S/backup_policy_network/MatMul:A2S/backup_policy_network/backup_policy_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
:A2S/backup_policy_network/LayerNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
(A2S/backup_policy_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
	container *
shape:
�
/A2S/backup_policy_network/LayerNorm/beta/AssignAssign(A2S/backup_policy_network/LayerNorm/beta:A2S/backup_policy_network/LayerNorm/beta/Initializer/zeros*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
�
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:
�
:A2S/backup_policy_network/LayerNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
)A2S/backup_policy_network/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma
�
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma
�
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma
�
BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
8A2S/backup_policy_network/LayerNorm/moments/StopGradientStopGradient0A2S/backup_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_policy_network/add8A2S/backup_policy_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
FA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
x
3A2S/backup_policy_network/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼�+*
dtype0
�
1A2S/backup_policy_network/LayerNorm/batchnorm/addAdd4A2S/backup_policy_network/LayerNorm/moments/variance3A2S/backup_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
3A2S/backup_policy_network/LayerNorm/batchnorm/RsqrtRsqrt1A2S/backup_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
1A2S/backup_policy_network/LayerNorm/batchnorm/mulMul3A2S/backup_policy_network/LayerNorm/batchnorm/Rsqrt.A2S/backup_policy_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_1MulA2S/backup_policy_network/add1A2S/backup_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2Mul0A2S/backup_policy_network/LayerNorm/moments/mean1A2S/backup_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
1A2S/backup_policy_network/LayerNorm/batchnorm/subSub-A2S/backup_policy_network/LayerNorm/beta/read3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
�
3A2S/backup_policy_network/LayerNorm/batchnorm/add_1Add3A2S/backup_policy_network/LayerNorm/batchnorm/mul_11A2S/backup_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
d
A2S/backup_policy_network/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *��?
�
A2S/backup_policy_network/mulMulA2S/backup_policy_network/mul/x3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
A2S/backup_policy_network/AbsAbs3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
f
!A2S/backup_policy_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_policy_network/mul_1Mul!A2S/backup_policy_network/mul_1/xA2S/backup_policy_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/backup_policy_network/add_1AddA2S/backup_policy_network/mulA2S/backup_policy_network/mul_1*
T0*'
_output_shapes
:���������
p
+A2S/backup_policy_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
VA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB"      *
dtype0
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
seed28
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
5A2S/backup_policy_network/backup_policy_network/out/w
VariableV2*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
<A2S/backup_policy_network/backup_policy_network/out/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/wPA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
�
:A2S/backup_policy_network/backup_policy_network/out/w/readIdentity5A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
�
GA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
5A2S/backup_policy_network/backup_policy_network/out/b
VariableV2*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(
�
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:
�
"A2S/backup_policy_network/MatMul_1MatMulA2S/backup_policy_network/add_1:A2S/backup_policy_network/backup_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_policy_network/add_2Add"A2S/backup_policy_network/MatMul_1:A2S/backup_policy_network/backup_policy_network/out/b/read*
T0*'
_output_shapes
:���������
�
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"      *
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

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
seed2H
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

:
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/fc0/w
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
�
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:
�
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(
�
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
8A2S/best_policy_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    
�
&A2S/best_policy_network/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container 
�
-A2S/best_policy_network/LayerNorm/beta/AssignAssign&A2S/best_policy_network/LayerNorm/beta8A2S/best_policy_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
8A2S/best_policy_network/LayerNorm/gamma/Initializer/onesConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
'A2S/best_policy_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:
�
.A2S/best_policy_network/LayerNorm/gamma/AssignAssign'A2S/best_policy_network/LayerNorm/gamma8A2S/best_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
,A2S/best_policy_network/LayerNorm/gamma/readIdentity'A2S/best_policy_network/LayerNorm/gamma*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
�
@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
.A2S/best_policy_network/LayerNorm/moments/meanMeanA2S/best_policy_network/add@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
6A2S/best_policy_network/LayerNorm/moments/StopGradientStopGradient.A2S/best_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
DA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
v
1A2S/best_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
/A2S/best_policy_network/LayerNorm/batchnorm/addAdd2A2S/best_policy_network/LayerNorm/moments/variance1A2S/best_policy_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtRsqrt/A2S/best_policy_network/LayerNorm/batchnorm/add*'
_output_shapes
:���������*
T0
�
/A2S/best_policy_network/LayerNorm/batchnorm/mulMul1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
1A2S/best_policy_network/LayerNorm/batchnorm/mul_1MulA2S/best_policy_network/add/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
1A2S/best_policy_network/LayerNorm/batchnorm/mul_2Mul.A2S/best_policy_network/LayerNorm/moments/mean/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
/A2S/best_policy_network/LayerNorm/batchnorm/subSub+A2S/best_policy_network/LayerNorm/beta/read1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
1A2S/best_policy_network/LayerNorm/batchnorm/add_1Add1A2S/best_policy_network/LayerNorm/batchnorm/mul_1/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
b
A2S/best_policy_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/best_policy_network/mulMulA2S/best_policy_network/mul/x1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
A2S/best_policy_network/AbsAbs1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
d
A2S/best_policy_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/best_policy_network/mul_1MulA2S/best_policy_network/mul_1/xA2S/best_policy_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/best_policy_network/add_1AddA2S/best_policy_network/mulA2S/best_policy_network/mul_1*'
_output_shapes
:���������*
T0
n
)A2S/best_policy_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      *
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
ZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2u*
dtype0*
_output_shapes

:*

seed
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes
: *
T0
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
�
1A2S/best_policy_network/best_policy_network/out/w
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    
�
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:
�
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
�
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/add_16A2S/best_policy_network/best_policy_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/out/b/read*'
_output_shapes
:���������*
T0
�
TA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
seed2�
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes
: *
T0
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
3A2S/backup_value_network/backup_value_network/fc0/w
VariableV2*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
:A2S/backup_value_network/backup_value_network/fc0/w/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/wNA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
8A2S/backup_value_network/backup_value_network/fc0/w/readIdentity3A2S/backup_value_network/backup_value_network/fc0/w*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
EA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zerosConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
3A2S/backup_value_network/backup_value_network/fc0/b
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b
�
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b
�
8A2S/backup_value_network/backup_value_network/fc0/b/readIdentity3A2S/backup_value_network/backup_value_network/fc0/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
_output_shapes
:
�
A2S/backup_value_network/MatMulMatMulA2S/observations8A2S/backup_value_network/backup_value_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/backup_value_network/addAddA2S/backup_value_network/MatMul8A2S/backup_value_network/backup_value_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
9A2S/backup_value_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
valueB*    
�
'A2S/backup_value_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
	container *
shape:
�
.A2S/backup_value_network/LayerNorm/beta/AssignAssign'A2S/backup_value_network/LayerNorm/beta9A2S/backup_value_network/LayerNorm/beta/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
�
,A2S/backup_value_network/LayerNorm/beta/readIdentity'A2S/backup_value_network/LayerNorm/beta*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
_output_shapes
:
�
9A2S/backup_value_network/LayerNorm/gamma/Initializer/onesConst*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
(A2S/backup_value_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:*
shared_name *;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
	container *
shape:
�
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:
�
AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
7A2S/backup_value_network/LayerNorm/moments/StopGradientStopGradient/A2S/backup_value_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_value_network/add7A2S/backup_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
EA2S/backup_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
w
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
0A2S/backup_value_network/LayerNorm/batchnorm/addAdd3A2S/backup_value_network/LayerNorm/moments/variance2A2S/backup_value_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/RsqrtRsqrt0A2S/backup_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
0A2S/backup_value_network/LayerNorm/batchnorm/mulMul2A2S/backup_value_network/LayerNorm/batchnorm/Rsqrt-A2S/backup_value_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
2A2S/backup_value_network/LayerNorm/batchnorm/mul_1MulA2S/backup_value_network/add0A2S/backup_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/mul_2Mul/A2S/backup_value_network/LayerNorm/moments/mean0A2S/backup_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
0A2S/backup_value_network/LayerNorm/batchnorm/subSub,A2S/backup_value_network/LayerNorm/beta/read2A2S/backup_value_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
�
2A2S/backup_value_network/LayerNorm/batchnorm/add_1Add2A2S/backup_value_network/LayerNorm/batchnorm/mul_10A2S/backup_value_network/LayerNorm/batchnorm/sub*'
_output_shapes
:���������*
T0
c
A2S/backup_value_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/backup_value_network/mulMulA2S/backup_value_network/mul/x2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/backup_value_network/AbsAbs2A2S/backup_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
e
 A2S/backup_value_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_value_network/mul_1Mul A2S/backup_value_network/mul_1/xA2S/backup_value_network/Abs*'
_output_shapes
:���������*
T0
�
A2S/backup_value_network/add_1AddA2S/backup_value_network/mulA2S/backup_value_network/mul_1*
T0*'
_output_shapes
:���������
o
*A2S/backup_value_network/dropout/keep_probConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
TA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *��̽
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *���=
�
\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
�
3A2S/backup_value_network/backup_value_network/out/w
VariableV2*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
validate_shape(*
_output_shapes

:
�
8A2S/backup_value_network/backup_value_network/out/w/readIdentity3A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
EA2S/backup_value_network/backup_value_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
valueB*    
�
3A2S/backup_value_network/backup_value_network/out/b
VariableV2*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(*
_output_shapes
:
�
8A2S/backup_value_network/backup_value_network/out/b/readIdentity3A2S/backup_value_network/backup_value_network/out/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
_output_shapes
:
�
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/backup_value_network/add_2Add!A2S/backup_value_network/MatMul_18A2S/backup_value_network/backup_value_network/out/b/read*'
_output_shapes
:���������*
T0
�
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"      *
dtype0
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
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  �?
�
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2�
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes
: 
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
�
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
�
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:
�
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0
�
/A2S/best_value_network/best_value_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:
�
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
�
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    
�
%A2S/best_value_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape:
�
,A2S/best_value_network/LayerNorm/beta/AssignAssign%A2S/best_value_network/LayerNorm/beta7A2S/best_value_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
*A2S/best_value_network/LayerNorm/beta/readIdentity%A2S/best_value_network/LayerNorm/beta*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
7A2S/best_value_network/LayerNorm/gamma/Initializer/onesConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
&A2S/best_value_network/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
�
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
+A2S/best_value_network/LayerNorm/gamma/readIdentity&A2S/best_value_network/LayerNorm/gamma*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:
�
?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
-A2S/best_value_network/LayerNorm/moments/meanMeanA2S/best_value_network/add?A2S/best_value_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
5A2S/best_value_network/LayerNorm/moments/StopGradientStopGradient-A2S/best_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
CA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
1A2S/best_value_network/LayerNorm/moments/varianceMean:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceCA2S/best_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
u
0A2S/best_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
.A2S/best_value_network/LayerNorm/batchnorm/addAdd1A2S/best_value_network/LayerNorm/moments/variance0A2S/best_value_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/RsqrtRsqrt.A2S/best_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
.A2S/best_value_network/LayerNorm/batchnorm/mulMul0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt+A2S/best_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_1MulA2S/best_value_network/add.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_2Mul-A2S/best_value_network/LayerNorm/moments/mean.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
.A2S/best_value_network/LayerNorm/batchnorm/subSub*A2S/best_value_network/LayerNorm/beta/read0A2S/best_value_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/add_1Add0A2S/best_value_network/LayerNorm/batchnorm/mul_1.A2S/best_value_network/LayerNorm/batchnorm/sub*'
_output_shapes
:���������*
T0
a
A2S/best_value_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/best_value_network/mulMulA2S/best_value_network/mul/x0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/best_value_network/AbsAbs0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
c
A2S/best_value_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/best_value_network/mul_1MulA2S/best_value_network/mul_1/xA2S/best_value_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/best_value_network/add_1AddA2S/best_value_network/mulA2S/best_value_network/mul_1*
T0*'
_output_shapes
:���������
m
(A2S/best_value_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"      *
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

:*

seed*
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

:
�
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
�
/A2S/best_value_network/best_value_network/out/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:
�
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
�
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:���������
b
A2S/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
A2S/ReshapeReshapeA2S/backup_policy_network/add_2A2S/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
d
A2S/Reshape_1/shapeConst*
_output_shapes
:*
valueB"����   *
dtype0
�
A2S/Reshape_1ReshapeA2S/best_policy_network/add_2A2S/Reshape_1/shape*
Tshape0*'
_output_shapes
:���������*
T0
N
	A2S/ConstConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
P
A2S/Const_1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Y
A2S/Normal/locIdentityA2S/Reshape*'
_output_shapes
:���������*
T0
H
A2S/Normal/scaleIdentity	A2S/Const*
T0*
_output_shapes
: 
]
A2S/Normal_1/locIdentityA2S/Reshape_1*'
_output_shapes
:���������*
T0
L
A2S/Normal_1/scaleIdentityA2S/Const_1*
_output_shapes
: *
T0
o
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
q
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
q
,A2S/KullbackLeibler/kl_normal_normal/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *   ?
j
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal_1/scale*
_output_shapes
: *
T0
j
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal/scale*
T0*
_output_shapes
: 
�
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 
�
(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal_1/locA2S/Normal/loc*'
_output_shapes
:���������*
T0
�
-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:���������
�
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 
�
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*'
_output_shapes
:���������*
T0
�
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*
T0*
_output_shapes
: 
~
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
_output_shapes
: *
T0
�
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
T0*
_output_shapes
: 
�
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
_output_shapes
: *
T0
�
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*
T0*'
_output_shapes
:���������
\
A2S/Const_2Const*
_output_shapes
:*
valueB"       *
dtype0
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
u
%A2S/Normal_2/batch_shape_tensor/ShapeShapeA2S/Normal_1/loc*
T0*
out_type0*
_output_shapes
:
j
'A2S/Normal_2/batch_shape_tensor/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
]
A2S/concat/values_0Const*
valueB:
*
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

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_2/batch_shape_tensor/BroadcastArgsA2S/concat/axis*
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
A2S/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :
������������������*
seed2�*

seed
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :
������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*4
_output_shapes"
 :
������������������*
T0
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*
T0*4
_output_shapes"
 :
������������������
h
A2S/addAddA2S/mulA2S/Normal_1/loc*
T0*4
_output_shapes"
 :
������������������
h
A2S/Reshape_2/shapeConst*!
valueB"����
      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*+
_output_shapes
:���������
*
T0*
Tshape0
S
A2S/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*
T0*
N*'
_output_shapes
:���������*

Tidx0
�
LA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/minConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
TA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes
: *
T0
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
FA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
�
+A2S/backup_q_network/backup_q_network/fc0/w
VariableV2*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
validate_shape(*
_output_shapes

:
�
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/backup_q_network/fc0/b
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
2A2S/backup_q_network/backup_q_network/fc0/b/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/b=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
validate_shape(*
_output_shapes
:
�
0A2S/backup_q_network/backup_q_network/fc0/b/readIdentity+A2S/backup_q_network/backup_q_network/fc0/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
_output_shapes
:
�
A2S/backup_q_network/MatMulMatMulA2S/concat_10A2S/backup_q_network/backup_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/backup_q_network/addAddA2S/backup_q_network/MatMul0A2S/backup_q_network/backup_q_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
5A2S/backup_q_network/LayerNorm/beta/Initializer/zerosConst*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
#A2S/backup_q_network/LayerNorm/beta
VariableV2*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
*A2S/backup_q_network/LayerNorm/beta/AssignAssign#A2S/backup_q_network/LayerNorm/beta5A2S/backup_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
(A2S/backup_q_network/LayerNorm/beta/readIdentity#A2S/backup_q_network/LayerNorm/beta*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
_output_shapes
:
�
5A2S/backup_q_network/LayerNorm/gamma/Initializer/onesConst*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
$A2S/backup_q_network/LayerNorm/gamma
VariableV2*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
+A2S/backup_q_network/LayerNorm/gamma/AssignAssign$A2S/backup_q_network/LayerNorm/gamma5A2S/backup_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:
�
=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
3A2S/backup_q_network/LayerNorm/moments/StopGradientStopGradient+A2S/backup_q_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_q_network/add3A2S/backup_q_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
s
.A2S/backup_q_network/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼�+*
dtype0
�
,A2S/backup_q_network/LayerNorm/batchnorm/addAdd/A2S/backup_q_network/LayerNorm/moments/variance.A2S/backup_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
.A2S/backup_q_network/LayerNorm/batchnorm/RsqrtRsqrt,A2S/backup_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
,A2S/backup_q_network/LayerNorm/batchnorm/mulMul.A2S/backup_q_network/LayerNorm/batchnorm/Rsqrt)A2S/backup_q_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
.A2S/backup_q_network/LayerNorm/batchnorm/mul_1MulA2S/backup_q_network/add,A2S/backup_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
.A2S/backup_q_network/LayerNorm/batchnorm/mul_2Mul+A2S/backup_q_network/LayerNorm/moments/mean,A2S/backup_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
,A2S/backup_q_network/LayerNorm/batchnorm/subSub(A2S/backup_q_network/LayerNorm/beta/read.A2S/backup_q_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
.A2S/backup_q_network/LayerNorm/batchnorm/add_1Add.A2S/backup_q_network/LayerNorm/batchnorm/mul_1,A2S/backup_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
_
A2S/backup_q_network/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *��?
�
A2S/backup_q_network/mulMulA2S/backup_q_network/mul/x.A2S/backup_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/backup_q_network/AbsAbs.A2S/backup_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
a
A2S/backup_q_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/backup_q_network/add_1AddA2S/backup_q_network/mulA2S/backup_q_network/mul_1*
T0*'
_output_shapes
:���������
k
&A2S/backup_q_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
LA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/minConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *��̽*
dtype0*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
TA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
�
FA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
�
+A2S/backup_q_network/backup_q_network/out/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
�
2A2S/backup_q_network/backup_q_network/out/w/AssignAssign+A2S/backup_q_network/backup_q_network/out/wFA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
�
0A2S/backup_q_network/backup_q_network/out/w/readIdentity+A2S/backup_q_network/backup_q_network/out/w*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
�
=A2S/backup_q_network/backup_q_network/out/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/backup_q_network/out/b
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
2A2S/backup_q_network/backup_q_network/out/b/AssignAssign+A2S/backup_q_network/backup_q_network/out/b=A2S/backup_q_network/backup_q_network/out/b/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
validate_shape(*
_output_shapes
:
�
0A2S/backup_q_network/backup_q_network/out/b/readIdentity+A2S/backup_q_network/backup_q_network/out/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
_output_shapes
:
�
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/backup_q_network/add_2AddA2S/backup_q_network/MatMul_10A2S/backup_q_network/backup_q_network/out/b/read*'
_output_shapes
:���������*
T0
�
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"      *
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
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:
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

:
�
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
�
'A2S/best_q_network/best_q_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:
�
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
�
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    
�
'A2S/best_q_network/best_q_network/fc0/b
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
�
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
�
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
3A2S/best_q_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    
�
!A2S/best_q_network/LayerNorm/beta
VariableV2*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
(A2S/best_q_network/LayerNorm/beta/AssignAssign!A2S/best_q_network/LayerNorm/beta3A2S/best_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
�
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
"A2S/best_q_network/LayerNorm/gamma
VariableV2*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
'A2S/best_q_network/LayerNorm/gamma/readIdentity"A2S/best_q_network/LayerNorm/gamma*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:
�
;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
�
1A2S/best_q_network/LayerNorm/moments/StopGradientStopGradient)A2S/best_q_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
6A2S/best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
q
,A2S/best_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
*A2S/best_q_network/LayerNorm/batchnorm/addAdd-A2S/best_q_network/LayerNorm/moments/variance,A2S/best_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
,A2S/best_q_network/LayerNorm/batchnorm/RsqrtRsqrt*A2S/best_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
*A2S/best_q_network/LayerNorm/batchnorm/mulMul,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/mul_1MulA2S/best_q_network/add*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/mul_2Mul)A2S/best_q_network/LayerNorm/moments/mean*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
*A2S/best_q_network/LayerNorm/batchnorm/subSub&A2S/best_q_network/LayerNorm/beta/read,A2S/best_q_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/add_1Add,A2S/best_q_network/LayerNorm/batchnorm/mul_1*A2S/best_q_network/LayerNorm/batchnorm/sub*'
_output_shapes
:���������*
T0
]
A2S/best_q_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/best_q_network/mulMulA2S/best_q_network/mul/x,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
}
A2S/best_q_network/AbsAbs,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
_
A2S/best_q_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/best_q_network/mul_1MulA2S/best_q_network/mul_1/xA2S/best_q_network/Abs*'
_output_shapes
:���������*
T0
�
A2S/best_q_network/add_1AddA2S/best_q_network/mulA2S/best_q_network/mul_1*
T0*'
_output_shapes
:���������
i
$A2S/best_q_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"      
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
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
seed2�*
dtype0*
_output_shapes

:*

seed
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

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
�
'A2S/best_q_network/best_q_network/out/w
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(
�
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
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
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
�
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/add_1,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:���������
}
%A2S/Normal_3/log_prob/standardize/subSubA2S/actionsA2S/Normal_1/loc*
T0*'
_output_shapes
:���������
�
)A2S/Normal_3/log_prob/standardize/truedivRealDiv%A2S/Normal_3/log_prob/standardize/subA2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
A2S/Normal_3/log_prob/SquareSquare)A2S/Normal_3/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
`
A2S/Normal_3/log_prob/mul/xConst*
_output_shapes
: *
valueB
 *   �*
dtype0
�
A2S/Normal_3/log_prob/mulMulA2S/Normal_3/log_prob/mul/xA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:���������
U
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
T0*
_output_shapes
: 
`
A2S/Normal_3/log_prob/add/xConst*
valueB
 *�?k?*
dtype0*
_output_shapes
: 
y
A2S/Normal_3/log_prob/addAddA2S/Normal_3/log_prob/add/xA2S/Normal_3/log_prob/Log*
_output_shapes
: *
T0
�
A2S/Normal_3/log_prob/subSubA2S/Normal_3/log_prob/mulA2S/Normal_3/log_prob/add*'
_output_shapes
:���������*
T0
[
A2S/NegNegA2S/Normal_3/log_prob/sub*
T0*'
_output_shapes
:���������
[
	A2S/mul_1MulA2S/NegA2S/advantages*
T0*'
_output_shapes
:���������
\
A2S/Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
m

A2S/Mean_1MeanA2S/advantagesA2S/Const_3*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
p
A2S/average_advantage/tagsConst*&
valueB BA2S/average_advantage*
dtype0*
_output_shapes
: 
o
A2S/average_advantageScalarSummaryA2S/average_advantage/tags
A2S/Mean_1*
T0*
_output_shapes
: 
\
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
h

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
t
A2S/policy_network_loss/tagsConst*(
valueB BA2S/policy_network_loss*
dtype0*
_output_shapes
: 
s
A2S/policy_network_lossScalarSummaryA2S/policy_network_loss/tags
A2S/Mean_2*
T0*
_output_shapes
: 
�
A2S/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add_2A2S/returns*'
_output_shapes
:���������*
T0
\
A2S/Const_5Const*
_output_shapes
:*
valueB"       *
dtype0
t

A2S/Mean_3MeanA2S/SquaredDifferenceA2S/Const_5*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
r
A2S/value_network_loss/tagsConst*'
valueB BA2S/value_network_loss*
dtype0*
_output_shapes
: 
q
A2S/value_network_lossScalarSummaryA2S/value_network_loss/tags
A2S/Mean_3*
_output_shapes
: *
T0
�
A2S/SquaredDifference_1SquaredDifferenceA2S/best_q_network/add_2A2S/returns*'
_output_shapes
:���������*
T0
\
A2S/Const_6Const*
valueB"       *
dtype0*
_output_shapes
:
v

A2S/Mean_4MeanA2S/SquaredDifference_1A2S/Const_6*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
j
A2S/q_network_loss/tagsConst*#
valueB BA2S/q_network_loss*
dtype0*
_output_shapes
: 
i
A2S/q_network_lossScalarSummaryA2S/q_network_loss/tags
A2S/Mean_4*
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
A2S/gradients/FillFillA2S/gradients/ShapeA2S/gradients/Const*
T0*
_output_shapes
: 
|
+A2S/gradients/A2S/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
%A2S/gradients/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
#A2S/gradients/A2S/Mean_2_grad/ShapeShape	A2S/mul_1*
T0*
out_type0*
_output_shapes
:
�
"A2S/gradients/A2S/Mean_2_grad/TileTile%A2S/gradients/A2S/Mean_2_grad/Reshape#A2S/gradients/A2S/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
n
%A2S/gradients/A2S/Mean_2_grad/Shape_1Shape	A2S/mul_1*
T0*
out_type0*
_output_shapes
:
h
%A2S/gradients/A2S/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
m
#A2S/gradients/A2S/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"A2S/gradients/A2S/Mean_2_grad/ProdProd%A2S/gradients/A2S/Mean_2_grad/Shape_1#A2S/gradients/A2S/Mean_2_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
o
%A2S/gradients/A2S/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
'A2S/gradients/A2S/Mean_2_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
%A2S/gradients/A2S/Mean_2_grad/MaximumMaximum$A2S/gradients/A2S/Mean_2_grad/Prod_1'A2S/gradients/A2S/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
�
&A2S/gradients/A2S/Mean_2_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_2_grad/Prod%A2S/gradients/A2S/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
"A2S/gradients/A2S/Mean_2_grad/CastCast&A2S/gradients/A2S/Mean_2_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
%A2S/gradients/A2S/Mean_2_grad/truedivRealDiv"A2S/gradients/A2S/Mean_2_grad/Tile"A2S/gradients/A2S/Mean_2_grad/Cast*'
_output_shapes
:���������*
T0
i
"A2S/gradients/A2S/mul_1_grad/ShapeShapeA2S/Neg*
T0*
out_type0*
_output_shapes
:
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
 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_2_grad/truedivA2S/advantages*
T0*'
_output_shapes
:���������
�
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1
�
5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape*'
_output_shapes
:���������
�
7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ShapeShapeA2S/Normal_3/log_prob/mul*
_output_shapes
:*
T0*
out_type0
w
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
BA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
�
6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
=A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape*'
_output_shapes
:���������
�
GA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1*
_output_shapes
: 
u
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1ShapeA2S/Normal_3/log_prob/Square*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:���������
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1MulA2S/Normal_3/log_prob/mul/xEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
=A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape
�
GA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
3A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/x)A2S/Normal_3/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
�
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul*'
_output_shapes
:���������*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_3/log_prob/standardize/sub*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
RA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1A2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
MA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1
�
UA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:���������
�
WA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
out_type0*
_output_shapes
:*
T0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal_1/loc*
T0*
out_type0*
_output_shapes
:
�
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1
�
QA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:���������
�
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:���������
�
&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
�
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/ShapeShape A2S/best_policy_network/MatMul_1*
_output_shapes
:*
T0*
out_type0
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
AA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
KA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
�
:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
DA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul=^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
�
LA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulE^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
NA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1E^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*O
_classE
CAloc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/ShapeShapeA2S/best_policy_network/mul*
_output_shapes
:*
T0*
out_type0
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1ShapeA2S/best_policy_network/mul_1*
_output_shapes
:*
T0*
out_type0
�
FA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
AA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape*'
_output_shapes
:���������
�
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1*'
_output_shapes
:���������
w
4A2S/gradients/A2S/best_policy_network/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
DA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/mul_grad/Shape6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2A2S/gradients/A2S/best_policy_network/mul_grad/mulMulIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1MulA2S/best_policy_network/mul/xIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_1Sum4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1FA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_16A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
?A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape9^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1
�
GA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/mul_grad/Reshape@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape
�
IA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*'
_output_shapes
:���������
y
6A2S/gradients/A2S/best_policy_network/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1ShapeA2S/best_policy_network/Abs*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulMulKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1A2S/best_policy_network/Abs*'
_output_shapes
:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
AA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1
�
3A2S/gradients/A2S/best_policy_network/Abs_grad/SignSign1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/best_policy_network/Abs_grad/mulMulKA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_13A2S/gradients/A2S/best_policy_network/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*
N*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_policy_network/add]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
Tshape0*
_output_shapes
:*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegNegHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape.A2S/best_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul.A2S/best_policy_network/LayerNorm/moments/mean]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:���������
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
A2S/gradients/AddN_1AddN_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtA2S/gradients/AddN_1*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape2A2S/best_policy_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumSumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeShape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addAddDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modFloorModIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeRangeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0
�
PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/FillFillMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill*
N*#
_output_shapes
:���������*
T0
�
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumSA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:���������*
T0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeReshape[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencySA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileTileMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeNA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape2A2S/best_policy_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1MaximumLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0
�
PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/CastCastPA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truedivRealDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Cast*'
_output_shapes
:���������*
T0
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape6A2S/best_policy_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
�
dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulUA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradientN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpW^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeS^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
gA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*i
_class_
][loc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
iA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*e
_class[
YWloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_policy_network/add*
out_type0*
_output_shapes
:*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addAdd@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modFloorModEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeRangeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/delta*

Tidx0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/FillFillIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0
�
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/MaximumMaximumOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/y*#
_output_shapes
:���������*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeReshape]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileTileIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape.A2S/best_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumHA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/CastCastLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truedivRealDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������
�
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/add_grad/ShapeShapeA2S/best_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/add_grad/Sum_16A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
?A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/add_grad/Reshape9^A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1
�
GA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/add_grad/Reshape@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape
�
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
_output_shapes
:*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1
�
8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
BA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul;^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1
�
JA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulC^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul
�
LA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1C^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1
�
A2S/beta1_power/initial_valueConst*
valueB
 *fff?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
�
A2S/beta1_power/readIdentityA2S/beta1_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/beta2_power/initial_valueConst*
valueB
 *w�?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container 
�
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power/readIdentityA2S/beta2_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
LA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/w/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
LA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/b/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
�
AA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0
�
/A2S/A2S/best_policy_network/LayerNorm/beta/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container 
�
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
�
4A2S/A2S/best_policy_network/LayerNorm/beta/Adam/readIdentity/A2S/A2S/best_policy_network/LayerNorm/beta/Adam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
�
CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape:
�
8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/AssignAssign1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
�
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/readIdentity1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
�
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:
�
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
�
DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1
VariableV2*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
�
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/readIdentity2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
�
LA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:
�
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/w/AdamLA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
LA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
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
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/w:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/b:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonIA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:
�
@A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam&A2S/best_policy_network/LayerNorm/beta/A2S/A2S/best_policy_network/LayerNorm/beta/Adam1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
�
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/w:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
use_nesterov( *
_output_shapes

:
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/b:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonKA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
�
A2S/AdamNoOpL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
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
-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
'A2S/gradients_1/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
%A2S/gradients_1/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
'A2S/gradients_1/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_1/A2S/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
o
%A2S/gradients_1/A2S/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_1/A2S/Mean_3_grad/ProdProd'A2S/gradients_1/A2S/Mean_3_grad/Shape_1%A2S/gradients_1/A2S/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
k
)A2S/gradients_1/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_1/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_3_grad/Prod_1)A2S/gradients_1/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
�
(A2S/gradients_1/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_3_grad/Prod'A2S/gradients_1/A2S/Mean_3_grad/Maximum*
_output_shapes
: *
T0
�
$A2S/gradients_1/A2S/Mean_3_grad/CastCast(A2S/gradients_1/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*'
_output_shapes
:���������*
T0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add_2*
_output_shapes
:*
T0*
out_type0
}
2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1ShapeA2S/returns*
out_type0*
_output_shapes
:*
T0
�
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*'
_output_shapes
:���������*
T0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/best_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_3_grad/truediv*'
_output_shapes
:���������*
T0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
_output_shapes
:*
T0*
out_type0
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
GA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
BA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
LA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulMatMulJA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1MatMulA2S/best_value_network/add_1JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
EA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_depsNoOp<^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul>^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1
�
MA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIdentity;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulF^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul
�
OA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1Identity=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1F^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
7A2S/gradients_1/A2S/best_value_network/add_1_grad/ShapeShapeA2S/best_value_network/mul*
T0*
out_type0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1ShapeA2S/best_value_network/mul_1*
out_type0*
_output_shapes
:*
T0
�
GA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
5A2S/gradients_1/A2S/best_value_network/add_1_grad/SumSumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_1SumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
BA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
LA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1
x
5A2S/gradients_1/A2S/best_value_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
EA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/mul_grad/mulMulJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_1/A2S/best_value_network/mul_grad/SumSum3A2S/gradients_1/A2S/best_value_network/mul_grad/mulEA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/mul_grad/Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1MulA2S/best_value_network/mul/xJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
@A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1
�
HA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape
�
JA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*'
_output_shapes
:���������
z
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1ShapeA2S/best_value_network/Abs*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulMulLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1A2S/best_value_network/Abs*
T0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1MulA2S/best_value_network/mul_1/xLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
BA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape*
_output_shapes
: 
�
LA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1
�
4A2S/gradients_1/A2S/best_value_network/Abs_grad/SignSign0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_1/A2S/best_value_network/Abs_grad/mulMulLA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_14A2S/gradients_1/A2S/best_value_network/Abs_grad/Sign*'
_output_shapes
:���������*
T0
�
A2S/gradients_1/AddNAddNJA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_13A2S/gradients_1/A2S/best_value_network/Abs_grad/mul*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*
N*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
Tshape0*
_output_shapes
:*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegNegIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-A2S/best_value_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-A2S/best_value_network/LayerNorm/moments/mean^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:���������
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients_1/AddN_1AddN`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0A2S/best_value_network/LayerNorm/batchnorm/RsqrtA2S/gradients_1/AddN_1*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeShape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addAddCA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modFloorModJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeRangeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0
�
QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*
N*#
_output_shapes
:���������*
T0
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/MaximumMaximumTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordivFloorDivLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1MaximumMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/CastCastQA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truedivRealDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
�
eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarConstO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulMulVA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradientO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegYA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpX^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeT^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
hA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshapea^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*j
_class`
^\loc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
jA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitySA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Nega^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_value_network/add*
out_type0*
_output_shapes
:*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addAdd?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modFloorModFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeRangeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/FillFillJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/MaximumMaximumPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
T0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeReshape^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3Shape-A2S/best_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1ProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumIA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/CastCastMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truedivRealDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������
�
A2S/gradients_1/AddN_2AddN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyhA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truediv*'
_output_shapes
:���������*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N
�
5A2S/gradients_1/A2S/best_value_network/add_grad/ShapeShapeA2S/best_value_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_17A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
@A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1
�
HA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape*'
_output_shapes
:���������
�
JA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1*
_output_shapes
:*
T0
�
9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
CA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul<^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1
�
KA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulD^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
MA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1D^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1*
_output_shapes

:
�
A2S/beta1_power_1/initial_valueConst*
valueB
 *fff?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0
�
A2S/beta2_power_1
VariableV2*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
JA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    
�
8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container 
�
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
=A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
�
LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:
�
AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/b/AdamJA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
�
=A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
�
LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container 
�
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
�
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
�
@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zerosConst*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0
�
.A2S/A2S/best_value_network/LayerNorm/beta/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape:
�
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1
VariableV2*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/AssignAssign0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
�
5A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/readIdentity0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1*
_output_shapes
:*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
AA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
/A2S/A2S/best_value_network/LayerNorm/gamma/Adam
VariableV2*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/AssignAssign/A2S/A2S/best_value_network/LayerNorm/gamma/AdamAA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
4A2S/A2S/best_value_network/LayerNorm/gamma/Adam/readIdentity/A2S/A2S/best_value_network/LayerNorm/gamma/Adam*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:
�
CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1
VariableV2*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:
�
JA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zerosConst*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    *
dtype0
�
8A2S/A2S/best_value_network/best_value_network/out/w/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
=A2S/A2S/best_value_network/best_value_network/out/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/w/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
�
LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:
�
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
�
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0
�
8A2S/A2S/best_value_network/best_value_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:
�
?A2S/A2S/best_value_network/best_value_network/out/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/b/AdamJA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
�
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
�
LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0
�
:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:
�
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
�
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
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
A2S/Adam_1/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
W
A2S/Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/b8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonJA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
use_nesterov( *
_output_shapes
:
�
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
�
BA2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&A2S/best_value_network/LayerNorm/gamma/A2S/A2S/best_value_network/LayerNorm/gamma/Adam1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/w8A2S/A2S/best_value_network/best_value_network/out/w/Adam:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(
�

A2S/Adam_1NoOpL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
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
A2S/gradients_2/FillFillA2S/gradients_2/ShapeA2S/gradients_2/Const*
_output_shapes
: *
T0
~
-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
'A2S/gradients_2/A2S/Mean_4_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
|
%A2S/gradients_2/A2S/Mean_4_grad/ShapeShapeA2S/SquaredDifference_1*
_output_shapes
:*
T0*
out_type0
�
$A2S/gradients_2/A2S/Mean_4_grad/TileTile'A2S/gradients_2/A2S/Mean_4_grad/Reshape%A2S/gradients_2/A2S/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
~
'A2S/gradients_2/A2S/Mean_4_grad/Shape_1ShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
j
'A2S/gradients_2/A2S/Mean_4_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%A2S/gradients_2/A2S/Mean_4_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients_2/A2S/Mean_4_grad/ProdProd'A2S/gradients_2/A2S/Mean_4_grad/Shape_1%A2S/gradients_2/A2S/Mean_4_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
'A2S/gradients_2/A2S/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
k
)A2S/gradients_2/A2S/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'A2S/gradients_2/A2S/Mean_4_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_4_grad/Prod_1)A2S/gradients_2/A2S/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0
�
(A2S/gradients_2/A2S/Mean_4_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_4_grad/Prod'A2S/gradients_2/A2S/Mean_4_grad/Maximum*
_output_shapes
: *
T0
�
$A2S/gradients_2/A2S/Mean_4_grad/CastCast(A2S/gradients_2/A2S/Mean_4_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_2/A2S/Mean_4_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_4_grad/Tile$A2S/gradients_2/A2S/Mean_4_grad/Cast*
T0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
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
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_4_grad/truediv*'
_output_shapes
:���������*
T0
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/best_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
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
:*

Tidx0*
	keep_dims( *
T0
�
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
EA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyIdentity4A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*G
_class=
;9loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape
�
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/ShapeShapeA2S/best_q_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
>A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape*'
_output_shapes
:���������
�
HA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1
�
7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1MatMulA2S/best_q_network/add_1FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
AA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_depsNoOp8^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul:^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1
�
IA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyIdentity7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulB^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul
�
KA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1Identity9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1B^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
3A2S/gradients_2/A2S/best_q_network/add_1_grad/ShapeShapeA2S/best_q_network/mul*
T0*
out_type0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1ShapeA2S/best_q_network/mul_1*
out_type0*
_output_shapes
:*
T0
�
CA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_1SumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape
�
HA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1
t
1A2S/gradients_2/A2S/best_q_network/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
AA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/A2S/gradients_2/A2S/best_q_network/mul_grad/mulMulFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
/A2S/gradients_2/A2S/best_q_network/mul_grad/SumSum/A2S/gradients_2/A2S/best_q_network/mul_grad/mulAA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
3A2S/gradients_2/A2S/best_q_network/mul_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/mul_grad/Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1MulA2S/best_q_network/mul/xFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
<A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape*
_output_shapes
: 
�
FA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
v
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1ShapeA2S/best_q_network/Abs*
T0*
out_type0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulMulHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1A2S/best_q_network/Abs*'
_output_shapes
:���������*
T0
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape
�
HA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/best_q_network/Abs_grad/SignSign,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
/A2S/gradients_2/A2S/best_q_network/Abs_grad/mulMulHA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_10A2S/gradients_2/A2S/best_q_network/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
A2S/gradients_2/AddNAddNFA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1/A2S/gradients_2/A2S/best_q_network/Abs_grad/mul*
N*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/sub*
out_type0*
_output_shapes
:*
T0
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/mul_2*
_output_shapes
:*
T0*
out_type0
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul)A2S/best_q_network/LayerNorm/moments/meanZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
A2S/gradients_2/AddN_1AddN\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad,A2S/best_q_network/LayerNorm/batchnorm/RsqrtXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1SumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeShape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addAdd?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modFloorModFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*
_output_shapes
:
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/FillFillJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill*
N*#
_output_shapes
:���������*
T0
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordivFloorDivHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeReshapeXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2Shape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3Shape-A2S/best_q_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*
dtype0
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/CastCastMA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truedivRealDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
�
aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeSA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulMulRA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradientK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1cA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/NegNegUA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpT^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeP^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
dA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
fA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
�
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addAdd;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modFloorModBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeRangeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/delta*

Tidx0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/FillFillFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill*
N*#
_output_shapes
:���������*
T0
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/MaximumMaximumLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordivFloorDivDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
out_type0*
_output_shapes
:*
T0
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1ProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/CastCastIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������
�
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/add_grad/ShapeShapeA2S/best_q_network/MatMul*
out_type0*
_output_shapes
:*
T0
}
3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/add_grad/Shape3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
/A2S/gradients_2/A2S/best_q_network/add_grad/SumSumA2S/gradients_2/AddN_2AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
<A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/add_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape*'
_output_shapes
:���������
�
FA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
?A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul8^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
�
GA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul*'
_output_shapes
:���������
�
IA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1*
_output_shapes

:
�
A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta1_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape: 
�
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/beta2_power_2/initial_valueConst*
valueB
 *w�?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
�
A2S/beta2_power_2
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(
�
A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
BA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container 
�
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/w/AdamBA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
5A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
�
DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
�
2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:
�
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
�
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:*
T0
�
BA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container 
�
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/b/AdamBA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
5A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
�
DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0
�
2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
�
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
�
<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0
�
*A2S/A2S/best_q_network/LayerNorm/beta/Adam
VariableV2*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
1A2S/A2S/best_q_network/LayerNorm/beta/Adam/AssignAssign*A2S/A2S/best_q_network/LayerNorm/beta/Adam<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
/A2S/A2S/best_q_network/LayerNorm/beta/Adam/readIdentity*A2S/A2S/best_q_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1
VariableV2*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
3A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/AssignAssign,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
1A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/readIdentity,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:*
T0
�
=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
+A2S/A2S/best_q_network/LayerNorm/gamma/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/AssignAssign+A2S/A2S/best_q_network/LayerNorm/gamma/Adam=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
0A2S/A2S/best_q_network/LayerNorm/gamma/Adam/readIdentity+A2S/A2S/best_q_network/LayerNorm/gamma/Adam*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:*
T0
�
?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container 
�
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/readIdentity-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1*
_output_shapes
:*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
BA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
0A2S/A2S/best_q_network/best_q_network/out/w/Adam
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/w/AdamBA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
�
5A2S/A2S/best_q_network/best_q_network/out/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/w/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
�
DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
BA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    
�
0A2S/A2S/best_q_network/best_q_network/out/b/Adam
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
7A2S/A2S/best_q_network/best_q_network/out/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/b/AdamBA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
5A2S/A2S/best_q_network/best_q_network/out/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/b/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
�
DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1
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
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
�
7A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
U
A2S/Adam_2/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
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
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( *
_output_shapes

:
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/b0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonFA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
use_nesterov( *
_output_shapes
:
�
=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
�
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/w0A2S/A2S/best_q_network/best_q_network/out/w/Adam2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/b0A2S/A2S/best_q_network/best_q_network/out/b/Adam2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonHA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
use_nesterov( *
_output_shapes
:
�
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
_output_shapes
: *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�

A2S/Adam_2NoOpD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1

A2S/group_depsNoOp

A2S/group_deps_1NoOp
�
A2S/Merge/MergeSummaryMergeSummaryA2S/klA2S/average_advantageA2S/policy_network_lossA2S/value_network_lossA2S/q_network_loss*
N*
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
T0""�
model_variables��
�
*A2S/backup_policy_network/LayerNorm/beta:0/A2S/backup_policy_network/LayerNorm/beta/Assign/A2S/backup_policy_network/LayerNorm/beta/read:0
�
+A2S/backup_policy_network/LayerNorm/gamma:00A2S/backup_policy_network/LayerNorm/gamma/Assign0A2S/backup_policy_network/LayerNorm/gamma/read:0
�
(A2S/best_policy_network/LayerNorm/beta:0-A2S/best_policy_network/LayerNorm/beta/Assign-A2S/best_policy_network/LayerNorm/beta/read:0
�
)A2S/best_policy_network/LayerNorm/gamma:0.A2S/best_policy_network/LayerNorm/gamma/Assign.A2S/best_policy_network/LayerNorm/gamma/read:0
�
)A2S/backup_value_network/LayerNorm/beta:0.A2S/backup_value_network/LayerNorm/beta/Assign.A2S/backup_value_network/LayerNorm/beta/read:0
�
*A2S/backup_value_network/LayerNorm/gamma:0/A2S/backup_value_network/LayerNorm/gamma/Assign/A2S/backup_value_network/LayerNorm/gamma/read:0
�
'A2S/best_value_network/LayerNorm/beta:0,A2S/best_value_network/LayerNorm/beta/Assign,A2S/best_value_network/LayerNorm/beta/read:0
�
(A2S/best_value_network/LayerNorm/gamma:0-A2S/best_value_network/LayerNorm/gamma/Assign-A2S/best_value_network/LayerNorm/gamma/read:0

%A2S/backup_q_network/LayerNorm/beta:0*A2S/backup_q_network/LayerNorm/beta/Assign*A2S/backup_q_network/LayerNorm/beta/read:0
�
&A2S/backup_q_network/LayerNorm/gamma:0+A2S/backup_q_network/LayerNorm/gamma/Assign+A2S/backup_q_network/LayerNorm/gamma/read:0
y
#A2S/best_q_network/LayerNorm/beta:0(A2S/best_q_network/LayerNorm/beta/Assign(A2S/best_q_network/LayerNorm/beta/read:0
|
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0"�
	summaries�
�
A2S/kl:0
A2S/average_advantage:0
A2S/policy_network_loss:0
A2S/value_network_loss:0
A2S/q_network_loss:0
A2S/average_reward_1:0"�,
trainable_variables�,�+
�
7A2S/backup_policy_network/backup_policy_network/fc0/w:0<A2S/backup_policy_network/backup_policy_network/fc0/w/Assign<A2S/backup_policy_network/backup_policy_network/fc0/w/read:0
�
7A2S/backup_policy_network/backup_policy_network/fc0/b:0<A2S/backup_policy_network/backup_policy_network/fc0/b/Assign<A2S/backup_policy_network/backup_policy_network/fc0/b/read:0
�
*A2S/backup_policy_network/LayerNorm/beta:0/A2S/backup_policy_network/LayerNorm/beta/Assign/A2S/backup_policy_network/LayerNorm/beta/read:0
�
+A2S/backup_policy_network/LayerNorm/gamma:00A2S/backup_policy_network/LayerNorm/gamma/Assign0A2S/backup_policy_network/LayerNorm/gamma/read:0
�
7A2S/backup_policy_network/backup_policy_network/out/w:0<A2S/backup_policy_network/backup_policy_network/out/w/Assign<A2S/backup_policy_network/backup_policy_network/out/w/read:0
�
7A2S/backup_policy_network/backup_policy_network/out/b:0<A2S/backup_policy_network/backup_policy_network/out/b/Assign<A2S/backup_policy_network/backup_policy_network/out/b/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0
�
(A2S/best_policy_network/LayerNorm/beta:0-A2S/best_policy_network/LayerNorm/beta/Assign-A2S/best_policy_network/LayerNorm/beta/read:0
�
)A2S/best_policy_network/LayerNorm/gamma:0.A2S/best_policy_network/LayerNorm/gamma/Assign.A2S/best_policy_network/LayerNorm/gamma/read:0
�
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
�
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
�
5A2S/backup_value_network/backup_value_network/fc0/w:0:A2S/backup_value_network/backup_value_network/fc0/w/Assign:A2S/backup_value_network/backup_value_network/fc0/w/read:0
�
5A2S/backup_value_network/backup_value_network/fc0/b:0:A2S/backup_value_network/backup_value_network/fc0/b/Assign:A2S/backup_value_network/backup_value_network/fc0/b/read:0
�
)A2S/backup_value_network/LayerNorm/beta:0.A2S/backup_value_network/LayerNorm/beta/Assign.A2S/backup_value_network/LayerNorm/beta/read:0
�
*A2S/backup_value_network/LayerNorm/gamma:0/A2S/backup_value_network/LayerNorm/gamma/Assign/A2S/backup_value_network/LayerNorm/gamma/read:0
�
5A2S/backup_value_network/backup_value_network/out/w:0:A2S/backup_value_network/backup_value_network/out/w/Assign:A2S/backup_value_network/backup_value_network/out/w/read:0
�
5A2S/backup_value_network/backup_value_network/out/b:0:A2S/backup_value_network/backup_value_network/out/b/Assign:A2S/backup_value_network/backup_value_network/out/b/read:0
�
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
�
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0
�
'A2S/best_value_network/LayerNorm/beta:0,A2S/best_value_network/LayerNorm/beta/Assign,A2S/best_value_network/LayerNorm/beta/read:0
�
(A2S/best_value_network/LayerNorm/gamma:0-A2S/best_value_network/LayerNorm/gamma/Assign-A2S/best_value_network/LayerNorm/gamma/read:0
�
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
�
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0
�
-A2S/backup_q_network/backup_q_network/fc0/w:02A2S/backup_q_network/backup_q_network/fc0/w/Assign2A2S/backup_q_network/backup_q_network/fc0/w/read:0
�
-A2S/backup_q_network/backup_q_network/fc0/b:02A2S/backup_q_network/backup_q_network/fc0/b/Assign2A2S/backup_q_network/backup_q_network/fc0/b/read:0

%A2S/backup_q_network/LayerNorm/beta:0*A2S/backup_q_network/LayerNorm/beta/Assign*A2S/backup_q_network/LayerNorm/beta/read:0
�
&A2S/backup_q_network/LayerNorm/gamma:0+A2S/backup_q_network/LayerNorm/gamma/Assign+A2S/backup_q_network/LayerNorm/gamma/read:0
�
-A2S/backup_q_network/backup_q_network/out/w:02A2S/backup_q_network/backup_q_network/out/w/Assign2A2S/backup_q_network/backup_q_network/out/w/read:0
�
-A2S/backup_q_network/backup_q_network/out/b:02A2S/backup_q_network/backup_q_network/out/b/Assign2A2S/backup_q_network/backup_q_network/out/b/read:0
�
)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0
�
)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0
y
#A2S/best_q_network/LayerNorm/beta:0(A2S/best_q_network/LayerNorm/beta/Assign(A2S/best_q_network/LayerNorm/beta/read:0
|
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0
�
)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0
�
)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0"0
train_op$
"
A2S/Adam

A2S/Adam_1

A2S/Adam_2"�b
	variables�b�b
�
7A2S/backup_policy_network/backup_policy_network/fc0/w:0<A2S/backup_policy_network/backup_policy_network/fc0/w/Assign<A2S/backup_policy_network/backup_policy_network/fc0/w/read:0
�
7A2S/backup_policy_network/backup_policy_network/fc0/b:0<A2S/backup_policy_network/backup_policy_network/fc0/b/Assign<A2S/backup_policy_network/backup_policy_network/fc0/b/read:0
�
*A2S/backup_policy_network/LayerNorm/beta:0/A2S/backup_policy_network/LayerNorm/beta/Assign/A2S/backup_policy_network/LayerNorm/beta/read:0
�
+A2S/backup_policy_network/LayerNorm/gamma:00A2S/backup_policy_network/LayerNorm/gamma/Assign0A2S/backup_policy_network/LayerNorm/gamma/read:0
�
7A2S/backup_policy_network/backup_policy_network/out/w:0<A2S/backup_policy_network/backup_policy_network/out/w/Assign<A2S/backup_policy_network/backup_policy_network/out/w/read:0
�
7A2S/backup_policy_network/backup_policy_network/out/b:0<A2S/backup_policy_network/backup_policy_network/out/b/Assign<A2S/backup_policy_network/backup_policy_network/out/b/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
�
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0
�
(A2S/best_policy_network/LayerNorm/beta:0-A2S/best_policy_network/LayerNorm/beta/Assign-A2S/best_policy_network/LayerNorm/beta/read:0
�
)A2S/best_policy_network/LayerNorm/gamma:0.A2S/best_policy_network/LayerNorm/gamma/Assign.A2S/best_policy_network/LayerNorm/gamma/read:0
�
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
�
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
�
5A2S/backup_value_network/backup_value_network/fc0/w:0:A2S/backup_value_network/backup_value_network/fc0/w/Assign:A2S/backup_value_network/backup_value_network/fc0/w/read:0
�
5A2S/backup_value_network/backup_value_network/fc0/b:0:A2S/backup_value_network/backup_value_network/fc0/b/Assign:A2S/backup_value_network/backup_value_network/fc0/b/read:0
�
)A2S/backup_value_network/LayerNorm/beta:0.A2S/backup_value_network/LayerNorm/beta/Assign.A2S/backup_value_network/LayerNorm/beta/read:0
�
*A2S/backup_value_network/LayerNorm/gamma:0/A2S/backup_value_network/LayerNorm/gamma/Assign/A2S/backup_value_network/LayerNorm/gamma/read:0
�
5A2S/backup_value_network/backup_value_network/out/w:0:A2S/backup_value_network/backup_value_network/out/w/Assign:A2S/backup_value_network/backup_value_network/out/w/read:0
�
5A2S/backup_value_network/backup_value_network/out/b:0:A2S/backup_value_network/backup_value_network/out/b/Assign:A2S/backup_value_network/backup_value_network/out/b/read:0
�
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
�
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0
�
'A2S/best_value_network/LayerNorm/beta:0,A2S/best_value_network/LayerNorm/beta/Assign,A2S/best_value_network/LayerNorm/beta/read:0
�
(A2S/best_value_network/LayerNorm/gamma:0-A2S/best_value_network/LayerNorm/gamma/Assign-A2S/best_value_network/LayerNorm/gamma/read:0
�
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
�
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0
�
-A2S/backup_q_network/backup_q_network/fc0/w:02A2S/backup_q_network/backup_q_network/fc0/w/Assign2A2S/backup_q_network/backup_q_network/fc0/w/read:0
�
-A2S/backup_q_network/backup_q_network/fc0/b:02A2S/backup_q_network/backup_q_network/fc0/b/Assign2A2S/backup_q_network/backup_q_network/fc0/b/read:0

%A2S/backup_q_network/LayerNorm/beta:0*A2S/backup_q_network/LayerNorm/beta/Assign*A2S/backup_q_network/LayerNorm/beta/read:0
�
&A2S/backup_q_network/LayerNorm/gamma:0+A2S/backup_q_network/LayerNorm/gamma/Assign+A2S/backup_q_network/LayerNorm/gamma/read:0
�
-A2S/backup_q_network/backup_q_network/out/w:02A2S/backup_q_network/backup_q_network/out/w/Assign2A2S/backup_q_network/backup_q_network/out/w/read:0
�
-A2S/backup_q_network/backup_q_network/out/b:02A2S/backup_q_network/backup_q_network/out/b/Assign2A2S/backup_q_network/backup_q_network/out/b/read:0
�
)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0
�
)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0
y
#A2S/best_q_network/LayerNorm/beta:0(A2S/best_q_network/LayerNorm/beta/Assign(A2S/best_q_network/LayerNorm/beta/read:0
|
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0
�
)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0
�
)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0
C
A2S/beta1_power:0A2S/beta1_power/AssignA2S/beta1_power/read:0
C
A2S/beta2_power:0A2S/beta2_power/AssignA2S/beta2_power/read:0
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam:0AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/read:0
�
>A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/read:0
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam:0AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/read:0
�
>A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/read:0
�
1A2S/A2S/best_policy_network/LayerNorm/beta/Adam:06A2S/A2S/best_policy_network/LayerNorm/beta/Adam/Assign6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/read:0
�
3A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1:08A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Assign8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/read:0
�
2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam:07A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Assign7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/read:0
�
4A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1:09A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Assign9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/read:0
�
<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam:0AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/read:0
�
>A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/read:0
�
<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam:0AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/read:0
�
>A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/read:0
I
A2S/beta1_power_1:0A2S/beta1_power_1/AssignA2S/beta1_power_1/read:0
I
A2S/beta2_power_1:0A2S/beta2_power_1/AssignA2S/beta2_power_1/read:0
�
:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:0?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Assign?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/read:0
�
<A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1:0AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/read:0
�
:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:0?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Assign?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/read:0
�
<A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1:0AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/read:0
�
0A2S/A2S/best_value_network/LayerNorm/beta/Adam:05A2S/A2S/best_value_network/LayerNorm/beta/Adam/Assign5A2S/A2S/best_value_network/LayerNorm/beta/Adam/read:0
�
2A2S/A2S/best_value_network/LayerNorm/beta/Adam_1:07A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Assign7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/read:0
�
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam:06A2S/A2S/best_value_network/LayerNorm/gamma/Adam/Assign6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/read:0
�
3A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1:08A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Assign8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/read:0
�
:A2S/A2S/best_value_network/best_value_network/out/w/Adam:0?A2S/A2S/best_value_network/best_value_network/out/w/Adam/Assign?A2S/A2S/best_value_network/best_value_network/out/w/Adam/read:0
�
<A2S/A2S/best_value_network/best_value_network/out/w/Adam_1:0AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/read:0
�
:A2S/A2S/best_value_network/best_value_network/out/b/Adam:0?A2S/A2S/best_value_network/best_value_network/out/b/Adam/Assign?A2S/A2S/best_value_network/best_value_network/out/b/Adam/read:0
�
<A2S/A2S/best_value_network/best_value_network/out/b/Adam_1:0AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/read:0
I
A2S/beta1_power_2:0A2S/beta1_power_2/AssignA2S/beta1_power_2/read:0
I
A2S/beta2_power_2:0A2S/beta2_power_2/AssignA2S/beta2_power_2/read:0
�
2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam:07A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Assign7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/read:0
�
4A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1:09A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/read:0
�
2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam:07A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Assign7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/read:0
�
4A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1:09A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/read:0
�
,A2S/A2S/best_q_network/LayerNorm/beta/Adam:01A2S/A2S/best_q_network/LayerNorm/beta/Adam/Assign1A2S/A2S/best_q_network/LayerNorm/beta/Adam/read:0
�
.A2S/A2S/best_q_network/LayerNorm/beta/Adam_1:03A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Assign3A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/read:0
�
-A2S/A2S/best_q_network/LayerNorm/gamma/Adam:02A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Assign2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/read:0
�
/A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1:04A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Assign4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/read:0
�
2A2S/A2S/best_q_network/best_q_network/out/w/Adam:07A2S/A2S/best_q_network/best_q_network/out/w/Adam/Assign7A2S/A2S/best_q_network/best_q_network/out/w/Adam/read:0
�
4A2S/A2S/best_q_network/best_q_network/out/w/Adam_1:09A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/read:0
�
2A2S/A2S/best_q_network/best_q_network/out/b/Adam:07A2S/A2S/best_q_network/best_q_network/out/b/Adam/Assign7A2S/A2S/best_q_network/best_q_network/out/b/Adam/read:0
�
4A2S/A2S/best_q_network/best_q_network/out/b/Adam_1:09A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/read:0�sn	(       �pJ	:��E�z�A*

A2S/average_reward_1   A �(       �pJ	!�E�z�A*

A2S/average_reward_1  0A�`�5(       �pJ	O'�E�z�A*

A2S/average_reward_1   A!W�(       �pJ		��E�z�A*

A2S/average_reward_1  $A��o(       �pJ	~HF�z�A*

A2S/average_reward_1��,A���(       �pJ	��F�z�A*

A2S/average_reward_1��*A9�}F(       �pJ	4uF�z�A*

A2S/average_reward_1  0AL���(       �pJ	�F�z�A*

A2S/average_reward_1  *A�G�(       �pJ	�1F�z�A*

A2S/average_reward_1UU%A!+X(       �pJ	�|#F�z�A*

A2S/average_reward_1  (AW�(       �pJ	D,F�z�A*

A2S/average_reward_1/�(Ai��(       �pJ	�1F�z�A*

A2S/average_reward_1  $AN��a(       �pJ	�:F�z�A*

A2S/average_reward_1vb'A̙��(       �pJ	��@F�z�A*

A2S/average_reward_1n�&Az�ɮ(       �pJ	/�FF�z�A*

A2S/average_reward_1ww'A�j(       �pJ	
^LF�z�A*

A2S/average_reward_1  %A��;h(       �pJ	)�QF�z�A*

A2S/average_reward_1��!A xU(       �pJ	�[F�z�A*

A2S/average_reward_1UU%A�c�1(       �pJ	gaF�z�A*

A2S/average_reward_1�%AZ�=(       �pJ	�hF�z�A*

A2S/average_reward_1��(A���(       �pJ	��lF�z�A*

A2S/average_reward_1b&A��k(       �pJ	]tF�z�A*

A2S/average_reward_1�E'A?s�(       �pJ	��zF�z�A*

A2S/average_reward_1��&AH���(       �pJ	<�F�z�A*

A2S/average_reward_1��&A�p(       �pJ	V�F�z�A*

A2S/average_reward_1�'A��O(       �pJ	�K�F�z�A*

A2S/average_reward_1vb'A��+�(       �pJ	E�F�z�A*

A2S/average_reward_1	�%A�K�(       �pJ	�|�F�z�A*

A2S/average_reward_1۶%A��bV(       �pJ	�ޡF�z�A*

A2S/average_reward_1�&A����(       �pJ	L��F�z�A*

A2S/average_reward_1��(As�z(       �pJ	oٱF�z�A*

A2S/average_reward_1SJ)A�B (       �pJ	|�F�z�A*

A2S/average_reward_1  )A�JH(       �pJ	�
�F�z�A*

A2S/average_reward_1��'Aa��(       �pJ		e�F�z�A*

A2S/average_reward_1��(A-�(       �pJ	X�F�z�A*

A2S/average_reward_1|�'A��c(       �pJ	��F�z�A*

A2S/average_reward_1  (A8�1�(       �pJ	�j�F�z�A*

A2S/average_reward_1��'Ah?R8(       �pJ	F��F�z�A*

A2S/average_reward_1��&A/��(       �pJ	3��F�z�A*

A2S/average_reward_1UU%A�b(       �pJ	��F�z�A*

A2S/average_reward_1ff&A4�0(       �pJ	�t�F�z�A*

A2S/average_reward_1p>&A�}'�(       �pJ	���F�z�A*

A2S/average_reward_1b&A�$��(       �pJ	&��F�z�A*

A2S/average_reward_1�'Aŧ<(       �pJ	�Y G�z�A*

A2S/average_reward_1  (AX�	f(       �pJ	G�z�A*

A2S/average_reward_1}�'Aw�{(       �pJ	�`G�z�A*

A2S/average_reward_1��'Av�f\(       �pJ	G�z�A*

A2S/average_reward_1��&Aד��(       �pJ	 G�z�A*

A2S/average_reward_1  &A���|(       �pJ	�G�z�A*

A2S/average_reward_1և&A�2�(       �pJ	�5$G�z�A*

A2S/average_reward_1ff&A	��(       �pJ	�+G�z�A*

A2S/average_reward_1��&A�	]\(       �pJ	R�1G�z�A*

A2S/average_reward_1��&A:9§(       �pJ	W}8G�z�A*

A2S/average_reward_1��&A|�;�(       �pJ	\�AG�z�A*

A2S/average_reward_1�K(A�R>W(       �pJ	��HG�z�A*

A2S/average_reward_1/�(A�}�(       �pJ	\�OG�z�A*

A2S/average_reward_1�$)Ad*��(       �pJ	�"UG�z�A*

A2S/average_reward_1��(A��(       �pJ	¡ZG�z�A*

A2S/average_reward_1��(A$=`U(       �pJ	�fbG�z�A*

A2S/average_reward_1�})A�#X+(       �pJ	QQfG�z�A*

A2S/average_reward_1��(A���3�       L	vM	��G�z�A�*�

A2S/kl��<

A2S/average_advantage��/�

A2S/policy_network_loss
�

A2S/value_network_lossm!?

A2S/q_network_loss䈗?�
<�+       ��K	� �G�z�A�*

A2S/average_reward_1�,Aa�ݙ+       ��K	���G�z�A�*

A2S/average_reward_1��2AX>�B+       ��K	"��G�z�A�*

A2S/average_reward_1��8AK�v+       ��K	���G�z�A�*

A2S/average_reward_1 �;Aq`7�+       ��K	��G�z�A�*

A2S/average_reward_1;AA���+       ��K	F�H�z�A�*

A2S/average_reward_1��CA)s<�+       ��K	��H�z�A�*

A2S/average_reward_1ïFAv��+       ��K	�%H�z�A�*

A2S/average_reward_1��JAY
��+       ��K	�?8H�z�A�*

A2S/average_reward_1��MA�s�g+       ��K	gHH�z�A�*

A2S/average_reward_1��PA.��+       ��K	�{XH�z�A�*

A2S/average_reward_1lTA�鐩+       ��K	�mfH�z�A�*

A2S/average_reward_1��VA�"�+       ��K	�^sH�z�A�*

A2S/average_reward_1�4YA�l;y+       ��K	ה�H�z�A�*

A2S/average_reward_1�\A�˽-+       ��K	M��H�z�A�*

A2S/average_reward_1t�`A�ሴ+       ��K	ߑ�H�z�A�*

A2S/average_reward_1��dAL��3+       ��K	���H�z�A�*

A2S/average_reward_1LlA�O��+       ��K	M��H�z�A�*

A2S/average_reward_1�[nA�?w+       ��K	�3�H�z�A�*

A2S/average_reward_17qA��_+       ��K	l� I�z�A�*

A2S/average_reward_1ffrAΥ�+       ��K	�I�z�A�*

A2S/average_reward_1UUuAq-�+       ��K	tq+I�z�A�*

A2S/average_reward_1�1xA{�+       ��K	Y�@I�z�A�*

A2S/average_reward_1��zAǔP�+       ��K	K�ZI�z�A�*

A2S/average_reward_1z�Ar�rJ+       ��K	jiI�z�A�*

A2S/average_reward_1���AL}�`+       ��K	!jyI�z�A�*

A2S/average_reward_1��A�E�+       ��K	�d�I�z�A�*

A2S/average_reward_1�قA"�5�+       ��K	��I�z�A�*

A2S/average_reward_1�E�Aר�.+       ��K	���I�z�A�*

A2S/average_reward_1�{�As8+       ��K	�I�z�A�*

A2S/average_reward_1ff�A�S�+       ��K	E��I�z�A�*

A2S/average_reward_1q�A���+       ��K	�C�I�z�A�*

A2S/average_reward_1���Ah�+�+       ��K	~��I�z�A�*

A2S/average_reward_1,�A���+       ��K	\�J�z�A�*

A2S/average_reward_1br�A{�{�+       ��K	J�z�A�*

A2S/average_reward_1K��A��,�+       ��K	7�-J�z�A�*

A2S/average_reward_1U��A|Oq+       ��K	�DJ�z�A�*

A2S/average_reward_1w��A���+       ��K	'
WJ�z�A�*

A2S/average_reward_1Dc�A��fo+       ��K	qlgJ�z�A�*

A2S/average_reward_1��A�F +       ��K	'N�J�z�A�*

A2S/average_reward_1q=�A`��+       ��K	w7�J�z�A�*

A2S/average_reward_1  �A�8�+       ��K	n�J�z�A�*

A2S/average_reward_1R��AQ�[b+       ��K	^g�J�z�A�*

A2S/average_reward_1�(�A���+       ��K	d��J�z�A�*

A2S/average_reward_133�A�/+       ��K	���J�z�A�*

A2S/average_reward_1�z�A����+       ��K	ֻ�J�z�A�*

A2S/average_reward_1�Q�A]J9+       ��K	
�J�z�A�*

A2S/average_reward_1)\�A�`�+       ��K	8�K�z�A�*

A2S/average_reward_1H�AUm��+       ��K	�G#K�z�A�*

A2S/average_reward_1�z�A8��g+       ��K	�8K�z�A�*

A2S/average_reward_1ff�Af�E�+       ��K	IQIK�z�A�*

A2S/average_reward_1�¡A��x%+       ��K	��ZK�z�A�*

A2S/average_reward_1��A�+VI+       ��K	R`nK�z�A�*

A2S/average_reward_1H�Ap��]+       ��K	l�K�z�A�*

A2S/average_reward_1�z�A�Igu+       ��K	e0�K�z�A�*

A2S/average_reward_1�Q�A;Rx+       ��K	T7�K�z�A�*

A2S/average_reward_1
שAs�zH+       ��K	/��K�z�A�*

A2S/average_reward_1�p�A8ʗ_+       ��K	N��K�z�A�*

A2S/average_reward_1ff�AS�+       ��K	x�K�z�A�*

A2S/average_reward_1)\�A}I3^+       ��K	���K�z�A�*

A2S/average_reward_1H�A�8"*�       L	vM	`�*L�z�A�*�

A2S/kl�P�=

A2S/average_advantage&�I�

A2S/policy_network_loss˟�

A2S/value_network_loss&hA

A2S/q_network_lossGA���+       ��K	�AL�z�A�*

A2S/average_reward_1{�AaLk+       ��K	/o\L�z�A�*

A2S/average_reward_133�AAX�=+       ��K	���L�z�A�*

A2S/average_reward_1{�A�{M�+       ��K	`\M�z�A�*

A2S/average_reward_1�p�A��@�+       ��K	��(M�z�A�*

A2S/average_reward_1\��A��+       ��K	]7M�z�A�*

A2S/average_reward_1��A07�+       ��K	�·M�z�A�*

A2S/average_reward_1�Q�A�b �+       ��K	:޾M�z�A�*

A2S/average_reward_1�Q�A��F�+       ��K	�C0N�z�A�*

A2S/average_reward_1\��A�U��+       ��K	�׌N�z�A�*

A2S/average_reward_1��AiG)�+       ��K	kM�N�z�A�*

A2S/average_reward_1��A}��+       ��K	�0�N�z�A�*

A2S/average_reward_1���Abm+       ��K	Y�O�z�A�*

A2S/average_reward_1R�B��	h+       ��K	j!,O�z�A�*

A2S/average_reward_1H�B�ҬT+       ��K	z:O�z�A�*

A2S/average_reward_1\�BDAF+       ��K	�DDO�z�A�*

A2S/average_reward_1��Bi�\[+       ��K	J�gO�z�A�*

A2S/average_reward_1��B�/�A+       ��K	��O�z�A�*

A2S/average_reward_1
�B�j�+       ��K	�ǹO�z�A�*

A2S/average_reward_1�
B๓�+       ��K	N�O�z�A�*

A2S/average_reward_1��
B~$N�+       ��K	���O�z�A�*

A2S/average_reward_1ףB'8��+       ��K	���O�z�A�*

A2S/average_reward_1�QB�x��+       ��K	�SP�z�A�*

A2S/average_reward_1{B2Y�+       ��K	�I)P�z�A�*

A2S/average_reward_1�B��!A+       ��K	�4DP�z�A�*

A2S/average_reward_1�B��Lv+       ��K	8�mP�z�A�*

A2S/average_reward_1
�B�<��+       ��K	r$�P�z�A�*

A2S/average_reward_1\�B�raG+       ��K	��P�z�A�*

A2S/average_reward_1�zB����+       ��K	K�%Q�z�A�*

A2S/average_reward_1\�B�z+       ��K	�GQ�z�A�*

A2S/average_reward_1�BT�;L+       ��K	��[Q�z�A�*

A2S/average_reward_1��B��)V+       ��K	!q�Q�z�A�*

A2S/average_reward_1ף$B�R�+       ��K	�m�Q�z�A�*

A2S/average_reward_1��&B�VS�+       ��K	f�&R�z�A�*

A2S/average_reward_1�(+Bܻ��+       ��K	��DR�z�A�*

A2S/average_reward_1�,Bqxa�+       ��K	jWSR�z�A�*

A2S/average_reward_1�-B���+       ��K	蜯R�z�A�*

A2S/average_reward_1�2Bj]K�+       ��K	TnS�z�A�*

A2S/average_reward_1�Q8B!��+       ��K	�OS�z�A�*

A2S/average_reward_1R�8B��~+       ��K	5x2S�z�A�*

A2S/average_reward_1H�9B��+       ��K	+J@S�z�A�*

A2S/average_reward_1��9B���+       ��K	U�aS�z�A�*

A2S/average_reward_1)\:Bm��?+       ��K	�E�S�z�A�*

A2S/average_reward_1�<By� +       ��K	� T�z�A�*

A2S/average_reward_1{BBa�Z�+       ��K	܌T�z�A�*

A2S/average_reward_1�GBBlo,�+       ��K	�)~T�z�A�*

A2S/average_reward_1��GB|2�b+       ��K	b��T�z�A�*

A2S/average_reward_1��HB;�V8+       ��K	\�T�z�A�*

A2S/average_reward_1q=KB}�U+       ��K	Q��T�z�A�*

A2S/average_reward_1�QKB.mqN+       ��K	L9U�z�A�*

A2S/average_reward_1ffMB+?+       ��K	�]nU�z�A�*

A2S/average_reward_1��RB4	^+       ��K	`|U�z�A�*

A2S/average_reward_1R�RB���p+       ��K	���U�z�A�*

A2S/average_reward_1�SB�/3�+       ��K	<��U�z�A�*

A2S/average_reward_1�QXB��+       ��K	x�V�z�A�*

A2S/average_reward_1��WB*�W�+       ��K	�_V�z�A�*

A2S/average_reward_1�pWB�Ӎ�+       ��K	�A,V�z�A�*

A2S/average_reward_1q=VBaj^S+       ��K	{�RV�z�A�*

A2S/average_reward_1\�WB��3�+       ��K	�o�V�z�A�*

A2S/average_reward_1R�YB����+       ��K	��V�z�A�*

A2S/average_reward_1q=[B�!e��       L	vM	�$�V�z�A�4*�

A2S/klҐl>

A2S/average_advantage�wZ�

A2S/policy_network_lossU��

A2S/value_network_loss�{�A

A2S/q_network_loss'��A6�>Z+       ��K	���V�z�A�4*

A2S/average_reward_1��ZB����+       ��K	W�z�A�4*

A2S/average_reward_1��[Bwz��+       ��K	��W�z�A�4*

A2S/average_reward_1\�[B�s��+       ��K	�)W�z�A�4*

A2S/average_reward_1H�ZBn��d+       ��K	��@W�z�A�4*

A2S/average_reward_133[B�{��+       ��K	^ eW�z�A�4*

A2S/average_reward_1q=\B�#+       ��K	I��W�z�A�4*

A2S/average_reward_1�]BQ���+       ��K	�H�W�z�A�4*

A2S/average_reward_1�^B8 +       ��K	|�W�z�A�4*

A2S/average_reward_1�p^B���-+       ��K	Ii�W�z�A�4*

A2S/average_reward_1�Q_B(�H+       ��K	d^�W�z�A�4*

A2S/average_reward_1�G_Bt���+       ��K	��X�z�A�4*

A2S/average_reward_1�z_B���3+       ��K	��X�z�A�4*

A2S/average_reward_1��_B�HR+       ��K	��!X�z�A�4*

A2S/average_reward_1ff_B��b+       ��K	�2X�z�A�4*

A2S/average_reward_1q=_B�r�+       ��K	��CX�z�A�4*

A2S/average_reward_1=
_B٧we+       ��K	��TX�z�A�4*

A2S/average_reward_1R�^B�Ǚ+       ��K	��}X�z�A�4*

A2S/average_reward_1)\`B��(+       ��K	��X�z�A�4*

A2S/average_reward_1H�`B�eέ+       ��K	�f�X�z�A�4*

A2S/average_reward_1H�_BE�Qt+       ��K	<ЮX�z�A�4*

A2S/average_reward_1�_BV��+       ��K	�m�X�z�A�4*

A2S/average_reward_133aB_���+       ��K	�� Y�z�A�4*

A2S/average_reward_1�bB�v�+       ��K	PY�z�A�4*

A2S/average_reward_1ףbB��0j+       ��K	Z*-Y�z�A�4*

A2S/average_reward_1H�bB]�-�+       ��K	ZKY�z�A�4*

A2S/average_reward_1�GcB���k+       ��K	��[Y�z�A�4*

A2S/average_reward_1�(bB�"+       ��K	n�kY�z�A�4*

A2S/average_reward_1  bB���.+       ��K	�{Y�z�A�4*

A2S/average_reward_1��aB����+       ��K	�c�Y�z�A�4*

A2S/average_reward_1��aB���+       ��K	�ĢY�z�A�4*

A2S/average_reward_1=
bB�l��+       ��K	���Y�z�A�4*

A2S/average_reward_1�zbB� ��+       ��K	><�Y�z�A�4*

A2S/average_reward_1�zdB�+       ��K	�jZ�z�A�4*

A2S/average_reward_1�dB"��+       ��K	�jZ�z�A�4*

A2S/average_reward_1\�dB>a
+       ��K	��/Z�z�A�4*

A2S/average_reward_1��dB.TK�+       ��K	~5?Z�z�A�4*

A2S/average_reward_1
�dB�é+       ��K	s�QZ�z�A�4*

A2S/average_reward_1=
eB`�0Q+       ��K	�]Z�z�A�4*

A2S/average_reward_1��dB��
b+       ��K		��Z�z�A�4*

A2S/average_reward_1��eB��+       ��K	��Z�z�A�4*

A2S/average_reward_1
�eB��m+       ��K	�̫Z�z�A�4*

A2S/average_reward_1R�dB�W�+       ��K	���Z�z�A�4*

A2S/average_reward_1��`Bܻ��+       ��K	;M�Z�z�A�4*

A2S/average_reward_1�z\BWV��+       ��K	p�	[�z�A�4*

A2S/average_reward_133]B��l+       ��K	5 [�z�A�4*

A2S/average_reward_1�]B/��+       ��K	��.[�z�A�4*

A2S/average_reward_1=
YBM��+       ��K	,�>[�z�A�4*

A2S/average_reward_1�VBH��3+       ��K	��h[�z�A�4*

A2S/average_reward_1=
RBf��3+       ��K	�Pr[�z�A�4*

A2S/average_reward_1�LB @�>+       ��K	NƗ[�z�A�4*

A2S/average_reward_1�MB���\+       ��K	��[�z�A�4*

A2S/average_reward_1��LB�&\�+       ��K	�ذ[�z�A�4*

A2S/average_reward_1�(HBz�`\+       ��K	�u�[�z�A�4*

A2S/average_reward_1�zGB=[�+       ��K	+�[�z�A�4*

A2S/average_reward_1ffGB�
f+       ��K	q��[�z�A�4*

A2S/average_reward_1R�GBR+*�+       ��K	>�\�z�A�4*

A2S/average_reward_1�(HB��k+       ��K	�k\�z�A�4*

A2S/average_reward_1��GB��c+       ��K	-�.\�z�A�4*

A2S/average_reward_1ffEB��/+       ��K	- :\�z�A�4*

A2S/average_reward_1
�DBN��8�       L	vM	��j\�z�A�D*�

A2S/kl�HA>

A2S/average_advantage�I��

A2S/policy_network_loss@~��

A2S/value_network_lossMM�A

A2S/q_network_loss֒�A��w!+       ��K	Tn\�z�A�D*

A2S/average_reward_1{DB�i��+       ��K	�
t\�z�A�D*

A2S/average_reward_1�CB�dr�+       ��K	btw\�z�A�D*

A2S/average_reward_1\�BB"+       ��K	ĩz\�z�A�D*

A2S/average_reward_1�G@B�[G�+       ��K	�ۂ\�z�A�D*

A2S/average_reward_1q=?B�&S�+       ��K	���\�z�A�D*

A2S/average_reward_133=B�q�+       ��K	;��\�z�A�D*

A2S/average_reward_1��:B����+       ��K	�Ș\�z�A�D*

A2S/average_reward_1ף8B.xB+       ��K	���\�z�A�D*

A2S/average_reward_1��2Bǖ�+       ��K	6�\�z�A�D*

A2S/average_reward_1�Q1Bw�+       ��K	L��\�z�A�D*

A2S/average_reward_1��0BP��Z+       ��K	��\�z�A�D*

A2S/average_reward_1
�+BZ��+       ��K	��\�z�A�D*

A2S/average_reward_1\�)BA���+       ��K	��\�z�A�D*

A2S/average_reward_1{%B���z+       ��K	���\�z�A�D*

A2S/average_reward_1R�#B(K'+       ��K	��\�z�A�D*

A2S/average_reward_1)\#B����+       ��K		�\�z�A�D*

A2S/average_reward_1{B��+       ��K	�C�\�z�A�D*

A2S/average_reward_1ffB�3�+       ��K	���\�z�A�D*

A2S/average_reward_1{B(�Je+       ��K	=D�\�z�A�D*

A2S/average_reward_1q=B�b�)+       ��K	|s]�z�A�D*

A2S/average_reward_1  B0='+       ��K	��]�z�A�D*

A2S/average_reward_1��B����+       ��K	��]�z�A�D*

A2S/average_reward_133BWT�c+       ��K	��]�z�A�D*

A2S/average_reward_1��B�F+       ��K	a� ]�z�A�D*

A2S/average_reward_1q=B��+       ��K	H-]�z�A�D*

A2S/average_reward_1�B�K�+       ��K	�}1]�z�A�D*

A2S/average_reward_1=
B�3�+       ��K	|!?]�z�A�D*

A2S/average_reward_1�QBe�+       ��K	j�C]�z�A�D*

A2S/average_reward_1�z B�j�+       ��K	�3L]�z�A�D*

A2S/average_reward_1���A�dӭ+       ��K	�Q]�z�A�D*

A2S/average_reward_1�G�A�5�U+       ��K	�`]�z�A�D*

A2S/average_reward_1)\�A��PJ+       ��K	��e]�z�A�D*

A2S/average_reward_1�G�A����+       ��K	�iu]�z�A�D*

A2S/average_reward_1�z�Ah_�+       ��K	76�]�z�A�D*

A2S/average_reward_1ף�A�k~]+       ��K	QE�]�z�A�D*

A2S/average_reward_1\��A��j+       ��K	���]�z�A�D*

A2S/average_reward_1��Af��+       ��K	)Ĥ]�z�A�D*

A2S/average_reward_1  �A�9vU+       ��K	��]�z�A�D*

A2S/average_reward_1
��A>�+       ��K	���]�z�A�D*

A2S/average_reward_1ff�Ao&�+       ��K	]�]�z�A�D*

A2S/average_reward_1��A�y
�+       ��K	�;�]�z�A�D*

A2S/average_reward_1���A���d+       ��K	���]�z�A�D*

A2S/average_reward_1�Q�A}�#J+       ��K	���]�z�A�D*

A2S/average_reward_1��A_U�+       ��K	Lu�]�z�A�D*

A2S/average_reward_1��A@�̼+       ��K	L��]�z�A�D*

A2S/average_reward_1{�AFm�+       ��K	���]�z�A�D*

A2S/average_reward_1���Ag��+       ��K	h��]�z�A�D*

A2S/average_reward_1�G�A�q�+       ��K	:~^�z�A�D*

A2S/average_reward_1��A�&Y+       ��K	<^�z�A�D*

A2S/average_reward_1
׿A�B+       ��K	�V^�z�A�D*

A2S/average_reward_1R��AW��+       ��K	�^�z�A�D*

A2S/average_reward_1=
�A'��+       ��K	��^�z�A�D*

A2S/average_reward_1���A���+       ��K	I*^�z�A�D*

A2S/average_reward_1=
�A�_�+       ��K	�d/^�z�A�D*

A2S/average_reward_1��AC��A+       ��K	�1<^�z�A�D*

A2S/average_reward_1)\�A*J�8+       ��K	l�F^�z�A�D*

A2S/average_reward_1H�A��ę+       ��K	U6T^�z�A�D*

A2S/average_reward_1�p�A�4��+       ��K	|�Y^�z�A�D*

A2S/average_reward_1��A��V+       ��K	��b^�z�A�D*

A2S/average_reward_1�̲A�iR+       ��K	�nf^�z�A�D*

A2S/average_reward_1��A�$x+       ��K	<�k^�z�A�D*

A2S/average_reward_1�G�A���|+       ��K	q^�z�A�D*

A2S/average_reward_1�G�Av�b#+       ��K	{u^�z�A�D*

A2S/average_reward_1��A�s�|+       ��K	,��^�z�A�D*

A2S/average_reward_1=
�A=/~V+       ��K	�+�^�z�A�D*

A2S/average_reward_1q=�A?�
�+       ��K	�~�^�z�A�D*

A2S/average_reward_1
ףA�h�6+       ��K	8d�^�z�A�D*

A2S/average_reward_1\��A#��g+       ��K	Mڛ^�z�A�D*

A2S/average_reward_1��Ar���+       ��K	Um�^�z�A�D*

A2S/average_reward_1{�A���c+       ��K	'��^�z�A�D*

A2S/average_reward_1�̞A�|�	+       ��K	���^�z�A�D*

A2S/average_reward_1=
�A��+       ��K	��^�z�A�D*

A2S/average_reward_1�G�Aka�R+       ��K	�O�^�z�A�D*

A2S/average_reward_1��A
edh+       ��K	i��^�z�A�D*

A2S/average_reward_1\��A���+       ��K	W�^�z�A�D*

A2S/average_reward_1H�A-~�+       ��K	���^�z�A�D*

A2S/average_reward_1  �A�xZ+       ��K	A��^�z�A�D*

A2S/average_reward_1ff�A��s+       ��K	"�^�z�A�D*

A2S/average_reward_1���A��CT+       ��K	��^�z�A�D*

A2S/average_reward_1�p�AB��@�       L	vM	�K5_�z�A�M*�

A2S/kl
�>

A2S/average_advantageD12>

A2S/policy_network_loss��X=

A2S/value_network_lossU�?

A2S/q_network_loss�Դ?h�`�+       ��K	� ;_�z�A�M*

A2S/average_reward_1=
�A�j2:+       ��K	.�@_�z�A�M*

A2S/average_reward_1�(�A�{�++       ��K	�F_�z�A�M*

A2S/average_reward_1)\�A�*�+       ��K	�lK_�z�A�M*

A2S/average_reward_1R��A�Io�+       ��K	²S_�z�A�M*

A2S/average_reward_1ff�A��+       ��K	s�\_�z�A�M*

A2S/average_reward_1�G}A~��+       ��K	�c_�z�A�M*

A2S/average_reward_1H�zAV.+       ��K	��i_�z�A�M*

A2S/average_reward_1��xA(��H+       ��K	p�o_�z�A�M*

A2S/average_reward_1{rAx�5�+       ��K	�pu_�z�A�M*

A2S/average_reward_1��qA��+       ��K	�]|_�z�A�M*

A2S/average_reward_1)\kAz�@+       ��K	�j�_�z�A�M*

A2S/average_reward_1{jAgz�Y+       ��K	q��_�z�A�M*

A2S/average_reward_1��hAp:,+       ��K	�;�_�z�A�M*

A2S/average_reward_1q=fA�Sey+       ��K	�L�_�z�A�M*

A2S/average_reward_1ףdA�'��+       ��K	��_�z�A�M*

A2S/average_reward_1ffbA�,��+       ��K	�$�_�z�A�M*

A2S/average_reward_1��XAnR�+       ��K	k��_�z�A�M*

A2S/average_reward_133WA�@+       ��K	���_�z�A�M*

A2S/average_reward_1
�SA�L+       ��K	��_�z�A�M*

A2S/average_reward_1ffRA����+       ��K	IR�_�z�A�M*

A2S/average_reward_1)\SA�r��+       ��K	��_�z�A�M*

A2S/average_reward_133SAf�9+       ��K	z�_�z�A�M*

A2S/average_reward_133SA=0v+       ��K	�I�_�z�A�M*

A2S/average_reward_1�SA�V��+       ��K	^K�_�z�A�M*

A2S/average_reward_1{RAV�+       ��K	�X�_�z�A�M*

A2S/average_reward_1��PA�O��+       ��K	}��_�z�A�M*

A2S/average_reward_1��PAF��(+       ��K	���_�z�A�M*

A2S/average_reward_1�QAʕ
�+       ��K	[P`�z�A�M*

A2S/average_reward_1ףPA���+       ��K	L�`�z�A�M*

A2S/average_reward_1�(PA��i+       ��K	@�`�z�A�M*

A2S/average_reward_1
�OAF�P+       ��K	��`�z�A�M*

A2S/average_reward_1ףPA!*P=+       ��K	WN`�z�A�M*

A2S/average_reward_1��PA�%t�+       ��K	��#`�z�A�M*

A2S/average_reward_1ףPAQ�7+       ��K	 �*`�z�A�M*

A2S/average_reward_1�QPA�}��+       ��K	V)1`�z�A�M*

A2S/average_reward_1H�NA����+       ��K	v8`�z�A�M*

A2S/average_reward_1��MA���+       ��K	Z=`�z�A�M*

A2S/average_reward_1��MA�Q�*+       ��K	��A`�z�A�M*

A2S/average_reward_1��LA�!�+       ��K	��G`�z�A�M*

A2S/average_reward_1�(LAcd<�+       ��K	?;N`�z�A�M*

A2S/average_reward_1)\KAY~�+       ��K	QX`�z�A�M*

A2S/average_reward_1{JA���+       ��K	��^`�z�A�M*

A2S/average_reward_1��IAuq�+       ��K	|Ad`�z�A�M*

A2S/average_reward_1��IAUuM+       ��K	��j`�z�A�M*

A2S/average_reward_1q=JAT�x+       ��K	��s`�z�A�M*

A2S/average_reward_1ףHA�A�+       ��K	��|`�z�A�M*

A2S/average_reward_1�GIA��+       ��K	G�`�z�A�M*

A2S/average_reward_133GAïYY+       ��K	�֊`�z�A�M*

A2S/average_reward_1
�GAݺ�	+       ��K	���`�z�A�M*

A2S/average_reward_1�GA�0`�+       ��K	&��`�z�A�M*

A2S/average_reward_1�GA~�+       ��K	���`�z�A�M*

A2S/average_reward_1��EA���X+       ��K	Mo�`�z�A�M*

A2S/average_reward_1q=FAeah+       ��K	���`�z�A�M*

A2S/average_reward_1�zDA_�<n+       ��K	���`�z�A�M*

A2S/average_reward_1��AA���+       ��K	2�`�z�A�M*

A2S/average_reward_1�?A��i�+       ��K	���`�z�A�M*

A2S/average_reward_1�Q@A/�+       ��K	#e�`�z�A�M*

A2S/average_reward_1
�?A�(Y+       ��K	
��`�z�A�M*

A2S/average_reward_1��@A���+       ��K	���`�z�A�M*

A2S/average_reward_1�?AV�+       ��K	kP�`�z�A�M*

A2S/average_reward_1�(@A:р+       ��K	���`�z�A�M*

A2S/average_reward_1\�>A�
+       ��K	�?�`�z�A�M*

A2S/average_reward_1��=A&\tX+       ��K	�� a�z�A�M*

A2S/average_reward_1R�>A�'U;+       ��K	y�a�z�A�M*

A2S/average_reward_1\�>AR\+       ��K	��a�z�A�M*

A2S/average_reward_1ff>A��p�+       ��K	�Ya�z�A�M*

A2S/average_reward_1��=A� >�+       ��K	�a�z�A�M*

A2S/average_reward_1�z<A�0!+       ��K	��&a�z�A�M*

A2S/average_reward_1ff:A0#V�+       ��K	��0a�z�A�M*

A2S/average_reward_1=
;A�9�
+       ��K	��7a�z�A�M*

A2S/average_reward_133;A�]�+       ��K	Sw>a�z�A�M*

A2S/average_reward_1�;A4ch+       ��K	�oCa�z�A�M*

A2S/average_reward_1
�;AV��+       ��K	EtKa�z�A�M*

A2S/average_reward_1=
;Ad��j+       ��K	L�Pa�z�A�M*

A2S/average_reward_133;A���+       ��K	��Ua�z�A�M*

A2S/average_reward_1�9A�m��+       ��K	С\a�z�A�M*

A2S/average_reward_1�(8A4k(+       ��K	��aa�z�A�M*

A2S/average_reward_1��5A�e@�+       ��K	>la�z�A�M*

A2S/average_reward_1R�6A�{υ+       ��K	ҽqa�z�A�M*

A2S/average_reward_1��5AR-F+       ��K	�za�z�A�M*

A2S/average_reward_1=
7A���+       ��K	T�~a�z�A�M*

A2S/average_reward_1H�6A���+       ��K	�c�a�z�A�M*

A2S/average_reward_1337A��"A+       ��K	��a�z�A�M*

A2S/average_reward_1�(8A�Dx+       ��K	b�a�z�A�M*

A2S/average_reward_1�p5A�>�+       ��K		w�a�z�A�M*

A2S/average_reward_1{6A�`�D+       ��K	�O�a�z�A�M*

A2S/average_reward_1��4A��i+       ��K	~��a�z�A�M*

A2S/average_reward_1��5A��+       ��K	s�a�z�A�M*

A2S/average_reward_1R�6A�˷+       ��K	Lݲa�z�A�M*

A2S/average_reward_1�G5A9}�+       ��K	��a�z�A�M*

A2S/average_reward_1�3A[�t�+       ��K	˨�a�z�A�M*

A2S/average_reward_1q=2Af�+       ��K	�|�a�z�A�M*

A2S/average_reward_1ff2A���+       ��K	�K�a�z�A�M*

A2S/average_reward_1333A_���+       ��K	�=�a�z�A�M*

A2S/average_reward_1�p1A����+       ��K	���a�z�A�M*

A2S/average_reward_1�Q0A���+       ��K	ڒ�a�z�A�M*

A2S/average_reward_1�(0A��db+       ��K	�4�a�z�A�M*

A2S/average_reward_1  0A
�^�+       ��K	m��a�z�A�M*

A2S/average_reward_1�z0A��J+       ��K	e��a�z�A�M*

A2S/average_reward_1�p1A���\�       L	vM	,�"b�z�A�U*�

A2S/kli3�=

A2S/average_advantage(���

A2S/policy_network_loss��9>

A2S/value_network_loss`K�?

A2S/q_network_loss�ӟ?!�\ +       ��K	a�>b�z�A�U*

A2S/average_reward_1�G9A Yn+       ��K	5�Db�z�A�U*

A2S/average_reward_1�9A��+       ��K	b�Rb�z�A�U*

A2S/average_reward_1�;Au�;+       ��K	7�bb�z�A�U*

A2S/average_reward_1\�>A)�L�+       ��K	�]qb�z�A�U*

A2S/average_reward_1  @A�K�+       ��K	��|b�z�A�U*

A2S/average_reward_1��@A�2�+       ��K	�ʂb�z�A�U*

A2S/average_reward_1�AA~�L+       ��K	��b�z�A�U*

A2S/average_reward_1{BA�`��+       ��K	�%�b�z�A�U*

A2S/average_reward_1�CA�-�+       ��K	�D�b�z�A�U*

A2S/average_reward_133GA�y+       ��K	]��b�z�A�U*

A2S/average_reward_1=
KAg�5�+       ��K	�h�b�z�A�U*

A2S/average_reward_1�OA�V�+       ��K	��b�z�A�U*

A2S/average_reward_133OA���+       ��K	���b�z�A�U*

A2S/average_reward_1
�OA���+       ��K	��c�z�A�U*

A2S/average_reward_133WAL�D�+       ��K	��#c�z�A�U*

A2S/average_reward_1�zXA ��k+       ��K	�+9c�z�A�U*

A2S/average_reward_1�[A���-+       ��K	�cZc�z�A�U*

A2S/average_reward_1)\cA0&+       ��K	�9fc�z�A�U*

A2S/average_reward_1�peA���+       ��K	���c�z�A�U*

A2S/average_reward_1q=rA)��+       ��K	&E�c�z�A�U*

A2S/average_reward_1��uAi��+       ��K	�7�c�z�A�U*

A2S/average_reward_1)\wA��+       ��K	Z�c�z�A�U*

A2S/average_reward_1�GyA��̉+       ��K	�"�c�z�A�U*

A2S/average_reward_1�p}A6�[V+       ��K	w�d�z�A�U*

A2S/average_reward_1=
�Ah%?G+       ��K	��d�z�A�U*

A2S/average_reward_1�p�A���U+       ��K	�d�z�A�U*

A2S/average_reward_1�G�Aꉽ�