       �K"	  �=�z�Abrain.Event:2纺�;     ��_�	�P�=�z�A"��
s
A2S/observationsPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
q
A2S/advantagesPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
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
dtype0*
_output_shapes
:*
shape:
n
A2S/returnsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
W
A2S/average_rewardPlaceholder*
shape:*
dtype0*
_output_shapes
:
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
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
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
:A2S/backup_policy_network/LayerNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
)A2S/backup_policy_network/LayerNorm/gamma
VariableV2*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:
�
BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
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
A2S/backup_policy_network/mulMulA2S/backup_policy_network/mul/x3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/backup_policy_network/AbsAbs3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
f
!A2S/backup_policy_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *���>
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
VA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
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
^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
seed28*
dtype0*
_output_shapes

:*

seed
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes
: 
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:*
T0
�
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:*
T0
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
VariableV2*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:
�
"A2S/backup_policy_network/MatMul_1MatMulA2S/backup_policy_network/add_1:A2S/backup_policy_network/backup_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/fc0/w
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
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    
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
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������
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
-A2S/best_policy_network/LayerNorm/beta/AssignAssign&A2S/best_policy_network/LayerNorm/beta8A2S/best_policy_network/LayerNorm/beta/Initializer/zeros*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
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
.A2S/best_policy_network/LayerNorm/gamma/AssignAssign'A2S/best_policy_network/LayerNorm/gamma8A2S/best_policy_network/LayerNorm/gamma/Initializer/ones*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
,A2S/best_policy_network/LayerNorm/gamma/readIdentity'A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
�
@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
.A2S/best_policy_network/LayerNorm/moments/meanMeanA2S/best_policy_network/add@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
6A2S/best_policy_network/LayerNorm/moments/StopGradientStopGradient.A2S/best_policy_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
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
2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
1A2S/best_policy_network/LayerNorm/batchnorm/mul_2Mul.A2S/best_policy_network/LayerNorm/moments/mean/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
A2S/best_policy_network/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *��?
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
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      
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
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *���=*
dtype0
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
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/out/w
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
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
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
TA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  ��*
dtype0
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
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
�
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:*
T0
�
3A2S/backup_value_network/backup_value_network/fc0/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
	container *
shape
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
8A2S/backup_value_network/backup_value_network/fc0/w/readIdentity3A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
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
VariableV2*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
A2S/backup_value_network/addAddA2S/backup_value_network/MatMul8A2S/backup_value_network/backup_value_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
9A2S/backup_value_network/LayerNorm/beta/Initializer/zerosConst*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
'A2S/backup_value_network/LayerNorm/beta
VariableV2*
shared_name *:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
.A2S/backup_value_network/LayerNorm/beta/AssignAssign'A2S/backup_value_network/LayerNorm/beta9A2S/backup_value_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
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
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(
�
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:
�
AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
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
<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_value_network/add7A2S/backup_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
EA2S/backup_value_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
w
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
0A2S/backup_value_network/LayerNorm/batchnorm/addAdd3A2S/backup_value_network/LayerNorm/moments/variance2A2S/backup_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
2A2S/backup_value_network/LayerNorm/batchnorm/RsqrtRsqrt0A2S/backup_value_network/LayerNorm/batchnorm/add*'
_output_shapes
:���������*
T0
�
0A2S/backup_value_network/LayerNorm/batchnorm/mulMul2A2S/backup_value_network/LayerNorm/batchnorm/Rsqrt-A2S/backup_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/mul_1MulA2S/backup_value_network/add0A2S/backup_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
A2S/backup_value_network/AbsAbs2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
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
*A2S/backup_value_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
TA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
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
dtype0*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2�
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
3A2S/backup_value_network/backup_value_network/out/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
	container *
shape
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
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(
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
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
/A2S/best_value_network/best_value_network/fc0/b
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
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
�
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
�
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
�
+A2S/best_value_network/LayerNorm/gamma/readIdentity&A2S/best_value_network/LayerNorm/gamma*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0
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
5A2S/best_value_network/LayerNorm/moments/StopGradientStopGradient-A2S/best_value_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
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
0A2S/best_value_network/LayerNorm/batchnorm/mul_1MulA2S/best_value_network/add.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_2Mul-A2S/best_value_network/LayerNorm/moments/mean.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
A2S/best_value_network/mulMulA2S/best_value_network/mul/x0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/best_value_network/AbsAbs0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
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
(A2S/best_value_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"      *
dtype0
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
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *���=*
dtype0
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
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
�
/A2S/best_value_network/best_value_network/out/w
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
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*'
_output_shapes
:���������*
T0
b
A2S/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
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
A2S/Reshape_1ReshapeA2S/best_policy_network/add_2A2S/Reshape_1/shape*'
_output_shapes
:���������*
T0*
Tshape0
N
	A2S/ConstConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
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
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0
�
(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal_1/locA2S/Normal/loc*
T0*'
_output_shapes
:���������
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
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
T0*
_output_shapes
: 
�
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
T0*
_output_shapes
: 
�
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
T0*
_output_shapes
: 
�
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*'
_output_shapes
:���������*
T0
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
A2S/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
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
A2S/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
]
A2S/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :������������������*
seed2�*

seed
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :������������������
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*
T0*4
_output_shapes"
 :������������������
h
A2S/addAddA2S/mulA2S/Normal_1/loc*4
_output_shapes"
 :������������������*
T0
h
A2S/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����      
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*+
_output_shapes
:���������*
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
N*'
_output_shapes
:���������*

Tidx0*
T0
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
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  �?*
dtype0
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
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes
: *
T0
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
�
FA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
+A2S/backup_q_network/backup_q_network/fc0/w
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
	container 
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
2A2S/backup_q_network/backup_q_network/fc0/b/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/b=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
validate_shape(
�
0A2S/backup_q_network/backup_q_network/fc0/b/readIdentity+A2S/backup_q_network/backup_q_network/fc0/b*
_output_shapes
:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma
�
+A2S/backup_q_network/LayerNorm/gamma/AssignAssign$A2S/backup_q_network/LayerNorm/gamma5A2S/backup_q_network/LayerNorm/gamma/Initializer/ones*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma
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
AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
.A2S/backup_q_network/LayerNorm/batchnorm/mul_1MulA2S/backup_q_network/add,A2S/backup_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
A2S/backup_q_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*'
_output_shapes
:���������*
T0
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
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *���=
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
VariableV2*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
0A2S/backup_q_network/backup_q_network/out/b/readIdentity+A2S/backup_q_network/backup_q_network/out/b*
_output_shapes
:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b
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
seed2�*
dtype0*
_output_shapes

:*

seed
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes
: 
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
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
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
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
3A2S/best_q_network/LayerNorm/beta/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
!A2S/best_q_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:
�
(A2S/best_q_network/LayerNorm/beta/AssignAssign!A2S/best_q_network/LayerNorm/beta3A2S/best_q_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*
_output_shapes
:*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
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
;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
q
,A2S/best_q_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
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
*A2S/best_q_network/LayerNorm/batchnorm/mulMul,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
,A2S/best_q_network/LayerNorm/batchnorm/mul_1MulA2S/best_q_network/add*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
,A2S/best_q_network/LayerNorm/batchnorm/mul_2Mul)A2S/best_q_network/LayerNorm/moments/mean*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
*A2S/best_q_network/LayerNorm/batchnorm/subSub&A2S/best_q_network/LayerNorm/beta/read,A2S/best_q_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
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
A2S/best_q_network/mulMulA2S/best_q_network/mul/x,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
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
A2S/best_q_network/mul_1MulA2S/best_q_network/mul_1/xA2S/best_q_network/Abs*'
_output_shapes
:���������*
T0
�
A2S/best_q_network/add_1AddA2S/best_q_network/mulA2S/best_q_network/mul_1*'
_output_shapes
:���������*
T0
i
$A2S/best_q_network/dropout/keep_probConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"      *
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
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
�
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
'A2S/best_q_network/best_q_network/out/w
VariableV2*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:
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
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
�
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
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
)A2S/Normal_3/log_prob/standardize/truedivRealDiv%A2S/Normal_3/log_prob/standardize/subA2S/Normal_1/scale*'
_output_shapes
:���������*
T0
�
A2S/Normal_3/log_prob/SquareSquare)A2S/Normal_3/log_prob/standardize/truediv*'
_output_shapes
:���������*
T0
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
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
_output_shapes
: *
T0
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
A2S/Normal_3/log_prob/subSubA2S/Normal_3/log_prob/mulA2S/Normal_3/log_prob/add*
T0*'
_output_shapes
:���������
[
A2S/NegNegA2S/Normal_3/log_prob/sub*'
_output_shapes
:���������*
T0
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

A2S/Mean_1MeanA2S/advantagesA2S/Const_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
A2S/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add_2A2S/returns*
T0*'
_output_shapes
:���������
\
A2S/Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
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
A2S/SquaredDifference_1SquaredDifferenceA2S/best_q_network/add_2A2S/returns*
T0*'
_output_shapes
:���������
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
A2S/q_network_loss/tagsConst*
dtype0*
_output_shapes
: *#
valueB BA2S/q_network_loss
i
A2S/q_network_lossScalarSummaryA2S/q_network_loss/tags
A2S/Mean_4*
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
"A2S/gradients/A2S/Mean_2_grad/ProdProd%A2S/gradients/A2S/Mean_2_grad/Shape_1#A2S/gradients/A2S/Mean_2_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
o
%A2S/gradients/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
'A2S/gradients/A2S/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
%A2S/gradients/A2S/Mean_2_grad/MaximumMaximum$A2S/gradients/A2S/Mean_2_grad/Prod_1'A2S/gradients/A2S/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
�
&A2S/gradients/A2S/Mean_2_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_2_grad/Prod%A2S/gradients/A2S/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
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
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_2_grad/truedivA2S/advantages*
T0*'
_output_shapes
:���������
�
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
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
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1ShapeA2S/Normal_3/log_prob/Square*
_output_shapes
:*
T0*
out_type0
�
BA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_3/log_prob/Square*'
_output_shapes
:���������*
T0
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1MulA2S/Normal_3/log_prob/mul/xEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
3A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/x)A2S/Normal_3/log_prob/standardize/truediv*'
_output_shapes
:���������*
T0
�
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul*'
_output_shapes
:���������*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_3/log_prob/standardize/sub*
_output_shapes
:*
T0*
out_type0
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
RA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1A2S/Normal_1/scale*'
_output_shapes
:���������*
T0
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
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal_1/scale*'
_output_shapes
:���������*
T0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2*'
_output_shapes
:���������*
T0
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
UA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape
�
WA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
T0*
out_type0*
_output_shapes
:
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
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
�
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/ShapeShape A2S/best_policy_network/MatMul_1*
_output_shapes
:*
T0*
out_type0
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
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
NA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1E^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:
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
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
AA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
w
4A2S/gradients/A2S/best_policy_network/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/mul_grad/Shape6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
6A2S/gradients/A2S/best_policy_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulMulKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1A2S/best_policy_network/Abs*
T0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
AA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape
�
KA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
3A2S/gradients/A2S/best_policy_network/Abs_grad/SignSign1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
2A2S/gradients/A2S/best_policy_network/Abs_grad/mulMulKA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_13A2S/gradients/A2S/best_policy_network/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*
N*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape.A2S/best_policy_network/LayerNorm/moments/mean*
out_type0*
_output_shapes
:*
T0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape
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
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������*
T0
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
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
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumSumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumSA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeReshape[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencySA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileTileMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeNA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truedivRealDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������
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
dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0
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
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modFloorModEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
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
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill*#
_output_shapes
:���������*
T0*
N
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeReshape]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileTileIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape.A2S/best_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
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
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������
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
DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/add_grad/Sum_16A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1*
_output_shapes
:*
T0
�
8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power/readIdentityA2S/beta1_power*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: *
T0
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
VariableV2*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1
VariableV2*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container *
shape
:*
dtype0
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
LA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    
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
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
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
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(
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
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
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
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    
�
0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam
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
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
�
DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0
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
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/readIdentity2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0
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
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container 
�
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/w/AdamLA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
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
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
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
VariableV2*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0
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
A2S/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
S
A2S/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
U
A2S/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
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
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
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
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: *
T0
�
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
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
valueB"      *
dtype0*
_output_shapes
:
�
'A2S/gradients_1/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
%A2S/gradients_1/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference*
T0*
out_type0*
_output_shapes
:
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
'A2S/gradients_1/A2S/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
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
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add_2*
_output_shapes
:*
T0*
out_type0
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
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*'
_output_shapes
:���������*
T0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/best_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
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
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
GA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
BA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape
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
=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1MatMulA2S/best_value_network/add_1JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
GA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
JA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape
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
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
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
HA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*'
_output_shapes
:���������
z
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1ShapeA2S/best_value_network/Abs*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
LA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
4A2S/gradients_1/A2S/best_value_network/Abs_grad/SignSign0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/Abs_grad/mulMulLA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_14A2S/gradients_1/A2S/best_value_network/Abs_grad/Sign*
T0*'
_output_shapes
:���������
�
A2S/gradients_1/AddNAddNJA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_13A2S/gradients_1/A2S/best_value_network/Abs_grad/mul*'
_output_shapes
:���������*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*
N
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
out_type0*
_output_shapes
:*
T0
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape
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
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
out_type0*
_output_shapes
:*
T0
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegNegIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-A2S/best_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
A2S/gradients_1/AddN_1AddN`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0A2S/best_value_network/LayerNorm/batchnorm/RsqrtA2S/gradients_1/AddN_1*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
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
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
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
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
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
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
�
eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegYA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
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
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addAdd?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeRangeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/FillFillJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill*
N*#
_output_shapes
:���������*
T0
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/MaximumMaximumPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/y*#
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
T0*
_output_shapes
:
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeReshape^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
out_type0*
_output_shapes
:*
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3Shape-A2S/best_value_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumIA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
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
A2S/gradients_1/AddN_2AddN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyhA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truediv*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������*
T0
�
5A2S/gradients_1/A2S/best_value_network/add_grad/ShapeShapeA2S/best_value_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
JA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*
_output_shapes
: *
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(
�
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
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
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
�
=A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
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
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    
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
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
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
VariableV2*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape:*
dtype0
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
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/AssignAssign/A2S/A2S/best_value_network/LayerNorm/gamma/AdamAA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(
�
4A2S/A2S/best_value_network/LayerNorm/gamma/Adam/readIdentity/A2S/A2S/best_value_network/LayerNorm/gamma/Adam*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
�
CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0
�
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1
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
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
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
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
=A2S/A2S/best_value_network/best_value_network/out/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/w/Adam*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:*
T0
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
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:*
T0
�
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
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
LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
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
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/w8A2S/A2S/best_value_network/best_value_network/out/w/Adam:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
use_nesterov( *
_output_shapes

:
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
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
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
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�

A2S/Adam_1NoOpL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
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
-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
'A2S/gradients_2/A2S/Mean_4_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
|
%A2S/gradients_2/A2S/Mean_4_grad/ShapeShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
�
$A2S/gradients_2/A2S/Mean_4_grad/TileTile'A2S/gradients_2/A2S/Mean_4_grad/Reshape%A2S/gradients_2/A2S/Mean_4_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
~
'A2S/gradients_2/A2S/Mean_4_grad/Shape_1ShapeA2S/SquaredDifference_1*
out_type0*
_output_shapes
:*
T0
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
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
_output_shapes
:*
T0*
out_type0

4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1ShapeA2S/returns*
T0*
out_type0*
_output_shapes
:
�
BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_4_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/best_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_4_grad/truediv*'
_output_shapes
:���������*
T0
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*'
_output_shapes
:���������*
T0
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/ShapeShapeA2S/best_q_network/MatMul_1*
_output_shapes
:*
T0*
out_type0

5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1MatMulA2S/best_q_network/add_1FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
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
1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
FA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape*'
_output_shapes
:���������
�
HA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1*'
_output_shapes
:���������
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
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
<A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape
�
FA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
v
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulMulHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1A2S/best_q_network/Abs*
T0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
>A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape*
_output_shapes
: 
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
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
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1
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
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*
T0*'
_output_shapes
:���������
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
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad,A2S/best_q_network/LayerNorm/batchnorm/RsqrtXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
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
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2Shape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3Shape-A2S/best_q_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0
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
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
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
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
out_type0*
_output_shapes
:*
T0
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
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modFloorModBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/FillFillFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:���������
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
out_type0*
_output_shapes
:*
T0
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
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
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
1A2S/gradients_2/A2S/best_q_network/add_grad/ShapeShapeA2S/best_q_network/MatMul*
T0*
out_type0*
_output_shapes
:
}
3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
<A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/add_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape
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
A2S/beta1_power_2/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0
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
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power_2/readIdentityA2S/beta1_power_2*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: *
T0
�
A2S/beta2_power_2/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/beta2_power_2
VariableV2*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
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
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
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
5A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:*
T0
�
DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:
�
9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(
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
VariableV2*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:
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
0A2S/A2S/best_q_network/LayerNorm/gamma/Adam/readIdentity+A2S/A2S/best_q_network/LayerNorm/gamma/Adam*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:
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
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/readIdentity-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:
�
BA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    
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
5A2S/A2S/best_q_network/best_q_network/out/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/w/Adam*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    
�
2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1
VariableV2*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:*
dtype0
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
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
�
BA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
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
7A2S/A2S/best_q_network/best_q_network/out/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/b/AdamBA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
�
5A2S/A2S/best_q_network/best_q_network/out/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/b/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
�
DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0
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
A2S/Adam_2/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
W
A2S/Adam_2/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
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
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/w0A2S/A2S/best_q_network/best_q_network/out/w/Adam2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
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
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
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
T0"I��]��     (+�E	��=�z�AJ��
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
A2S/actionsPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
q
A2S/advantagesPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
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
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
�
PA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
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
GA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
valueB*    
�
5A2S/backup_policy_network/backup_policy_network/fc0/b
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b
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
VariableV2*
shared_name *;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
/A2S/backup_policy_network/LayerNorm/beta/AssignAssign(A2S/backup_policy_network/LayerNorm/beta:A2S/backup_policy_network/LayerNorm/beta/Initializer/zeros*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:
�
:A2S/backup_policy_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
valueB*  �?
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
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:
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
=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_policy_network/add8A2S/backup_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
FA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
A2S/backup_policy_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
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
VA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
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
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:
�
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b
�
"A2S/backup_policy_network/MatMul_1MatMulA2S/backup_policy_network/add_1:A2S/backup_policy_network/backup_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes
: *
T0
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/fc0/w
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
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
�
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:*
T0
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
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
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
8A2S/best_policy_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0
�
&A2S/best_policy_network/LayerNorm/beta
VariableV2*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
.A2S/best_policy_network/LayerNorm/gamma/AssignAssign'A2S/best_policy_network/LayerNorm/gamma8A2S/best_policy_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
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
DA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
v
1A2S/best_policy_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
�
/A2S/best_policy_network/LayerNorm/batchnorm/addAdd2A2S/best_policy_network/LayerNorm/moments/variance1A2S/best_policy_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
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
1A2S/best_policy_network/LayerNorm/batchnorm/mul_1MulA2S/best_policy_network/add/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
1A2S/best_policy_network/LayerNorm/batchnorm/mul_2Mul.A2S/best_policy_network/LayerNorm/moments/mean/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
A2S/best_policy_network/mul_1MulA2S/best_policy_network/mul_1/xA2S/best_policy_network/Abs*'
_output_shapes
:���������*
T0
�
A2S/best_policy_network/add_1AddA2S/best_policy_network/mulA2S/best_policy_network/mul_1*'
_output_shapes
:���������*
T0
n
)A2S/best_policy_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *��̽*
dtype0
�
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *���=
�
ZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*
seed2u*
dtype0*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
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

:
�
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
�
1A2S/best_policy_network/best_policy_network/out/w
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
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(
�
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
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
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  ��*
dtype0
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
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:*
T0
�
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:*
T0
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
EA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
valueB*    
�
3A2S/backup_value_network/backup_value_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
	container *
shape:
�
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
VariableV2*
shared_name *:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
.A2S/backup_value_network/LayerNorm/beta/AssignAssign'A2S/backup_value_network/LayerNorm/beta9A2S/backup_value_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta
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
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma
�
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma
�
AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
7A2S/backup_value_network/LayerNorm/moments/StopGradientStopGradient/A2S/backup_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
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
0A2S/backup_value_network/LayerNorm/batchnorm/mulMul2A2S/backup_value_network/LayerNorm/batchnorm/Rsqrt-A2S/backup_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
2A2S/backup_value_network/LayerNorm/batchnorm/mul_1MulA2S/backup_value_network/add0A2S/backup_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *���=*
dtype0
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
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: 
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
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
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
�
8A2S/backup_value_network/backup_value_network/out/w/readIdentity3A2S/backup_value_network/backup_value_network/out/w*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:*
T0
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
	container *
shape:
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
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"      
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
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:*

seed
�
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
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
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    
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
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
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
VariableV2*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0
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
+A2S/best_value_network/LayerNorm/gamma/readIdentity&A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
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
5A2S/best_value_network/LayerNorm/moments/StopGradientStopGradient-A2S/best_value_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
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
.A2S/best_value_network/LayerNorm/batchnorm/addAdd1A2S/best_value_network/LayerNorm/moments/variance0A2S/best_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/RsqrtRsqrt.A2S/best_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
.A2S/best_value_network/LayerNorm/batchnorm/mulMul0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt+A2S/best_value_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_1MulA2S/best_value_network/add.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
0A2S/best_value_network/LayerNorm/batchnorm/mul_2Mul-A2S/best_value_network/LayerNorm/moments/mean.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
.A2S/best_value_network/LayerNorm/batchnorm/subSub*A2S/best_value_network/LayerNorm/beta/read0A2S/best_value_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/add_1Add0A2S/best_value_network/LayerNorm/batchnorm/mul_1.A2S/best_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
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
A2S/best_value_network/mul_1MulA2S/best_value_network/mul_1/xA2S/best_value_network/Abs*'
_output_shapes
:���������*
T0
�
A2S/best_value_network/add_1AddA2S/best_value_network/mulA2S/best_value_network/mul_1*
T0*'
_output_shapes
:���������
m
(A2S/best_value_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
_output_shapes

:*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2�*
dtype0
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
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*'
_output_shapes
:���������*
T0
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
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
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
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal/scale*
T0*
_output_shapes
: 
�
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0
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
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:���������
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
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
T0*
_output_shapes
: 
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
valueB"       *
dtype0*
_output_shapes
:
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
R
A2S/kl/tagsConst*
_output_shapes
: *
valueB BA2S/kl*
dtype0
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
'A2S/Normal_2/batch_shape_tensor/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
]
A2S/concat/values_0Const*
valueB:*
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
A2S/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
dtype0*4
_output_shapes"
 :������������������*
seed2�*

seed*
T0
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*4
_output_shapes"
 :������������������*
T0
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*4
_output_shapes"
 :������������������*
T0
h
A2S/addAddA2S/mulA2S/Normal_1/loc*4
_output_shapes"
 :������������������*
T0
h
A2S/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����      
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*+
_output_shapes
:���������*
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
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  �?*
dtype0
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
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
	container 
�
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
�
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
�
=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
valueB*    
�
+A2S/backup_q_network/backup_q_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
	container *
shape:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
	container *
shape:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
	container *
shape:
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
3A2S/backup_q_network/LayerNorm/moments/StopGradientStopGradient+A2S/backup_q_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
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
/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
s
.A2S/backup_q_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
�
,A2S/backup_q_network/LayerNorm/batchnorm/addAdd/A2S/backup_q_network/LayerNorm/moments/variance.A2S/backup_q_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:���������
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
,A2S/backup_q_network/LayerNorm/batchnorm/subSub(A2S/backup_q_network/LayerNorm/beta/read.A2S/backup_q_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
�
.A2S/backup_q_network/LayerNorm/batchnorm/add_1Add.A2S/backup_q_network/LayerNorm/batchnorm/mul_1,A2S/backup_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
_
A2S/backup_q_network/mul/xConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
A2S/backup_q_network/mulMulA2S/backup_q_network/mul/x.A2S/backup_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
�
A2S/backup_q_network/AbsAbs.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
a
A2S/backup_q_network/mul_1/xConst*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*'
_output_shapes
:���������*
T0
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
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
	container 
�
2A2S/backup_q_network/backup_q_network/out/w/AssignAssign+A2S/backup_q_network/backup_q_network/out/wFA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
VariableV2*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_q_network/add_2AddA2S/backup_q_network/MatMul_10A2S/backup_q_network/backup_q_network/out/b/read*'
_output_shapes
:���������*
T0
�
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"      
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
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:*
T0
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
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
�
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
�
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
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
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(
�
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
�
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
3A2S/best_q_network/LayerNorm/beta/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
!A2S/best_q_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:
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
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:*
T0
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
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
,A2S/best_q_network/LayerNorm/batchnorm/RsqrtRsqrt*A2S/best_q_network/LayerNorm/batchnorm/add*'
_output_shapes
:���������*
T0
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
A2S/best_q_network/add_1AddA2S/best_q_network/mulA2S/best_q_network/mul_1*'
_output_shapes
:���������*
T0
i
$A2S/best_q_network/dropout/keep_probConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"      *
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

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
seed2�*
dtype0*
_output_shapes

:
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes
: 
�
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/sub*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
�
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
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
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
�
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
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
A2S/Normal_3/log_prob/mul/xConst*
valueB
 *   �*
dtype0*
_output_shapes
: 
�
A2S/Normal_3/log_prob/mulMulA2S/Normal_3/log_prob/mul/xA2S/Normal_3/log_prob/Square*'
_output_shapes
:���������*
T0
U
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
_output_shapes
: *
T0
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
A2S/NegNegA2S/Normal_3/log_prob/sub*'
_output_shapes
:���������*
T0
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
A2S/Mean_1*
_output_shapes
: *
T0
\
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
h

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
A2S/policy_network_loss/tagsConst*(
valueB BA2S/policy_network_loss*
dtype0*
_output_shapes
: 
s
A2S/policy_network_lossScalarSummaryA2S/policy_network_loss/tags
A2S/Mean_2*
_output_shapes
: *
T0
�
A2S/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add_2A2S/returns*
T0*'
_output_shapes
:���������
\
A2S/Const_5Const*
dtype0*
_output_shapes
:*
valueB"       
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
#A2S/gradients/A2S/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
'A2S/gradients/A2S/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
"A2S/gradients/A2S/Mean_2_grad/CastCast&A2S/gradients/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
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
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_2_grad/truediv*'
_output_shapes
:���������*
T0
�
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
:���������*
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
:���������*
T0
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
BA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
=A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape*'
_output_shapes
:���������*
T0
�
GA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
u
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1ShapeA2S/Normal_3/log_prob/Square*
_output_shapes
:*
T0*
out_type0
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
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
=A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
�
EA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape*
_output_shapes
: *
T0
�
GA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
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
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
UA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape
�
WA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
T0*
out_type0*
_output_shapes
:
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal_1/loc*
_output_shapes
:*
T0*
out_type0
�
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
T0*
out_type0*
_output_shapes
:
�
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/ShapeShape A2S/best_policy_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
AA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1
�
IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape
�
KA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
�
:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
NA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1E^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:
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
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
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
IA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1*'
_output_shapes
:���������
w
4A2S/gradients/A2S/best_policy_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
DA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/mul_grad/Shape6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_1Sum4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1FA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_16A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
IA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1
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
4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulMulKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1A2S/best_policy_network/Abs*
T0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/sub*
_output_shapes
:*
T0*
out_type0
�
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
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
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_policy_network/add]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������*
T0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegNegHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
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
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul.A2S/best_policy_network/LayerNorm/moments/mean]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtA2S/gradients/AddN_1*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
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
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumSumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeShape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumSA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeReshape[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencySA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileTileMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeNA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truedivRealDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape6A2S/best_policy_network/LayerNorm/moments/StopGradient*
out_type0*
_output_shapes
:*
T0
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
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
iA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*e
_class[
YWloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
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
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modFloorModEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeRangeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum*
T0*
_output_shapes
:
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
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumHA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/CastCastLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truedivRealDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������
�
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/add_grad/ShapeShapeA2S/best_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/add_grad/Sum_16A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
A2S/beta2_power/readIdentityA2S/beta2_power*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
LA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    
�
:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam
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
VariableV2*
dtype0*
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container *
shape
:
�
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
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
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
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
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
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
VariableV2*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape:*
dtype0
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/AssignAssign1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/readIdentity1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
�
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    
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
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
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
CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
�
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
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
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
�
CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
�
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/w:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
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
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
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
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
A2S/AdamNoOpL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
X
A2S/gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
A2S/gradients_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
k
A2S/gradients_1/FillFillA2S/gradients_1/ShapeA2S/gradients_1/Const*
T0*
_output_shapes
: 
~
-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
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
'A2S/gradients_1/A2S/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
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
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
$A2S/gradients_1/A2S/Mean_3_grad/CastCast(A2S/gradients_1/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add_2*
T0*
out_type0*
_output_shapes
:
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
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*'
_output_shapes
:���������*
T0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/best_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
CA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIdentity2A2S/gradients_1/A2S/SquaredDifference_grad/Reshape<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*E
_class;
97loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Reshape
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
GA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
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
MA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIdentity;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulF^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������*
T0
�
OA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1Identity=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1F^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1
�
7A2S/gradients_1/A2S/best_value_network/add_1_grad/ShapeShapeA2S/best_value_network/mul*
T0*
out_type0*
_output_shapes
:
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
;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
EA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1MulA2S/best_value_network/mul/xJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
JA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*'
_output_shapes
:���������
z
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
out_type0*
_output_shapes
:*
T0
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
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-A2S/best_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-A2S/best_value_network/LayerNorm/moments/mean^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
A2S/gradients_1/AddN_1AddN`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0A2S/best_value_network/LayerNorm/batchnorm/RsqrtA2S/gradients_1/AddN_1*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeShape1A2S/best_value_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*
N*#
_output_shapes
:���������*
T0
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/MaximumMaximumTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truedivRealDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Cast*'
_output_shapes
:���������*
T0
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
VA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarConstO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulMulVA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
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
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
jA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitySA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Nega^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
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
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modFloorModFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/FillFillJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill*
N*#
_output_shapes
:���������*
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeReshape^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1ProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
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
5A2S/gradients_1/A2S/best_value_network/add_grad/ShapeShapeA2S/best_value_network/MatMul*
T0*
out_type0*
_output_shapes
:
�
7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
JA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
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
MA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1D^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1*
_output_shapes

:
�
A2S/beta1_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0
�
A2S/beta1_power_1
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
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
=A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
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
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam
VariableV2*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:*
dtype0
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
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
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
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
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
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container 
�
7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/AssignAssign0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
5A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/readIdentity0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:*
T0
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
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container 
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
4A2S/A2S/best_value_network/LayerNorm/gamma/Adam/readIdentity/A2S/A2S/best_value_network/LayerNorm/gamma/Adam*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0
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
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0
�
JA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    
�
8A2S/A2S/best_value_network/best_value_network/out/w/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container 
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
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
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_value_network/best_value_network/out/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
8A2S/A2S/best_value_network/best_value_network/out/b/Adam
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
?A2S/A2S/best_value_network/best_value_network/out/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/b/AdamJA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
�
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
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
A2S/Adam_1/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/b8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonJA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/w8A2S/A2S/best_value_network/best_value_network/out/w/Adam:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
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
A2S/gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
A2S/gradients_2/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
%A2S/gradients_2/A2S/Mean_4_grad/ShapeShapeA2S/SquaredDifference_1*
_output_shapes
:*
T0*
out_type0
�
$A2S/gradients_2/A2S/Mean_4_grad/TileTile'A2S/gradients_2/A2S/Mean_4_grad/Reshape%A2S/gradients_2/A2S/Mean_4_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
_output_shapes
:*
T0*
out_type0
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
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_4_grad/truediv*
T0*'
_output_shapes
:���������
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/best_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
T0*'
_output_shapes
:���������
�
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*'
_output_shapes
:���������*
T0
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
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
3A2S/gradients_2/A2S/best_q_network/add_1_grad/ShapeShapeA2S/best_q_network/mul*
T0*
out_type0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1ShapeA2S/best_q_network/mul_1*
T0*
out_type0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
FA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape*'
_output_shapes
:���������
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
3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
/A2S/gradients_2/A2S/best_q_network/mul_grad/SumSum/A2S/gradients_2/A2S/best_q_network/mul_grad/mulAA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
<A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
�
DA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape*
_output_shapes
: *
T0
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
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1ShapeA2S/best_q_network/Abs*
T0*
out_type0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulMulHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1A2S/best_q_network/Abs*
T0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
�
FA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape*
_output_shapes
: 
�
HA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
�
0A2S/gradients_2/A2S/best_q_network/Abs_grad/SignSign,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
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
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:���������*
T0
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: *
T0
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeShape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
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
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:���������*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*
T0*0
_output_shapes
:������������������*

Tmultiples0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2Shape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0
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
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
�
aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeSA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
UA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/NegNegUA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
fA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*b
_classX
VTloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
out_type0*
_output_shapes
:*
T0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/FillFillFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
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
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:���������*
T0
�
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*'
_output_shapes
:���������*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N
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
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
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
A2S/beta2_power_2/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/beta2_power_2
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
BA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    
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
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/w/AdamBA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
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
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
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
DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
�
2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:
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
<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    
�
*A2S/A2S/best_q_network/LayerNorm/beta/Adam
VariableV2*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
1A2S/A2S/best_q_network/LayerNorm/beta/Adam/AssignAssign*A2S/A2S/best_q_network/LayerNorm/beta/Adam<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zeros*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
/A2S/A2S/best_q_network/LayerNorm/beta/Adam/readIdentity*A2S/A2S/best_q_network/LayerNorm/beta/Adam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
�
>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0
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
1A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/readIdentity,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1*
_output_shapes
:*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    
�
+A2S/A2S/best_q_network/LayerNorm/gamma/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:
�
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/AssignAssign+A2S/A2S/best_q_network/LayerNorm/gamma/Adam=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zeros*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
0A2S/A2S/best_q_network/LayerNorm/gamma/Adam/readIdentity+A2S/A2S/best_q_network/LayerNorm/gamma/Adam*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:*
T0
�
?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0
�
-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1
VariableV2*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0
�
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
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
VariableV2*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:*
dtype0
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
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
�
BA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
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
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/b0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonFA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( 
�
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
use_nesterov( 
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/w0A2S/A2S/best_q_network/best_q_network/out/w/Adam2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
use_nesterov( 
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
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
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
A2S/average_reward_1ScalarSummaryA2S/average_reward_1/tagsA2S/average_reward*
T0*
_output_shapes
: ""�
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
4A2S/A2S/best_q_network/best_q_network/out/b/Adam_1:09A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/read:0"�
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
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0P��(       �pJ	Yi�=�z�A*

A2S/average_reward_1  �A�6�(       �pJ	��=�z�A*

A2S/average_reward_1  �A��`D(       �pJ	a��=�z�A*

A2S/average_reward_1UUeAu}+(       �pJ	J�=�z�A*

A2S/average_reward_1  TA���(       �pJ	���=�z�A*

A2S/average_reward_1  pA�0a(       �pJ	ʜ�=�z�A*

A2S/average_reward_1���A�0t(       �pJ	>-�=�z�A*

A2S/average_reward_1۶}A��(       �pJ	׋�=�z�A*

A2S/average_reward_1  rA��E(       �pJ	�A�=�z�A*

A2S/average_reward_1UUeA�OI((       �pJ	�E�=�z�A*

A2S/average_reward_133cATs��(       �pJ	��=�z�A*

A2S/average_reward_1�[A#���(       �pJ	J��=�z�A*

A2S/average_reward_1��VAeUΡ(       �pJ	9��=�z�A*

A2S/average_reward_1�[A�?�*(       �pJ	]>�=�z�A*

A2S/average_reward_1I�TAS�Y�(       �pJ	���=�z�A*

A2S/average_reward_1""RA����(       �pJ	�� >�z�A*

A2S/average_reward_1  PA>��+(       �pJ	��>�z�A*

A2S/average_reward_1��SAkn�(       �pJ	��	>�z�A*

A2S/average_reward_1  PA"e��(       �pJ	��>�z�A*

A2S/average_reward_1l(OAI��h(       �pJ	xH>�z�A*

A2S/average_reward_1��MA���(       �pJ	Ⱥ>�z�A*

A2S/average_reward_1�<OAy�a(       �pJ	�>�z�A*

A2S/average_reward_1FMA8�?(       �pJ	�	>�z�A*

A2S/average_reward_1�7MA{
��(       �pJ	�u>�z�A*

A2S/average_reward_1UUMA���(       �pJ	�H#>�z�A*

A2S/average_reward_1R�NAH��O(       �pJ	�&>�z�A*

A2S/average_reward_1ىMA�~N(       �pJ	Er*>�z�A*

A2S/average_reward_1  PA���(       �pJ	`�->�z�A*

A2S/average_reward_1�mOA�|��(       �pJ	Ӱ1>�z�A*

A2S/average_reward_1�rOA���(       �pJ	��4>�z�A*

A2S/average_reward_1ffNAzչ(       �pJ	C7>�z�A*

A2S/average_reward_1cLArHD(       �pJ	�.;>�z�A*

A2S/average_reward_1 �MA[&](       �pJ	�=>�z�A*

A2S/average_reward_1�KA�0�{(       �pJ	�@@>�z�A*

A2S/average_reward_1ZZJAq&�(       �pJ	FFE>�z�A*

A2S/average_reward_1�AMA��qW(       �pJ	�H>�z�A*

A2S/average_reward_1��LAf�k(       �pJ	�J>�z�A*

A2S/average_reward_1��JA��*%(       �pJ	 �M>�z�A*

A2S/average_reward_1�JA�y�(       �pJ	�P>�z�A*

A2S/average_reward_1��IA��(       �pJ	d�R>�z�A*

A2S/average_reward_1  HAfLq(       �pJ	��U>�z�A*

A2S/average_reward_1%jGA���(       �pJ	z�Z>�z�A*

A2S/average_reward_1�HA��$�(       �pJ	++^>�z�A*

A2S/average_reward_1�/HA/�0 (       �pJ	��a>�z�A*

A2S/average_reward_1]HAH-"�(       �pJ	�>d>�z�A*

A2S/average_reward_1rGA��Z�(       �pJ	xXg>�z�A*

A2S/average_reward_1��FAe��G(       �pJ	'�k>�z�A*

A2S/average_reward_1�+HA��8�(       �pJ	��n>�z�A*

A2S/average_reward_1��GAA{��(       �pJ	�]q>�z�A*

A2S/average_reward_1n�FA��l�(       �pJ	5t>�z�A*

A2S/average_reward_1{FA��S(       �pJ	�
w>�z�A*

A2S/average_reward_1��EAO?��(       �pJ	�y>�z�A*

A2S/average_reward_1��DA���1(       �pJ	��|>�z�A*

A2S/average_reward_1�!EA��b�(       �pJ	�Q�>�z�A*

A2S/average_reward_1{	EAҙ�(       �pJ	�w�>�z�A*

A2S/average_reward_1��DA���(       �pJ	���>�z�A*

A2S/average_reward_1I�DAݪr�(       �pJ	Y��>�z�A*

A2S/average_reward_1��FAs�o?(       �pJ	�V�>�z�A*

A2S/average_reward_1	�EA�O�(       �pJ	���>�z�A*

A2S/average_reward_1��DA`]b(       �pJ	�_�>�z�A*

A2S/average_reward_1��EA3?�       L	vM	`+�>�z�A�*�

A2S/kl޺�<

A2S/average_advantager���

A2S/policy_network_loss���

A2S/value_network_loss�P�?

A2S/q_network_loss��?��f+       ��K	��>�z�A�*

A2S/average_reward_1��FA���+       ��K	e|�>�z�A�*

A2S/average_reward_1  HA�`�c+       ��K	���>�z�A�*

A2S/average_reward_1~�GA�I�+       ��K	��>�z�A�*

A2S/average_reward_1 �IA��
�+       ��K	���>�z�A�*

A2S/average_reward_1��KA>W��+       ��K	��>�z�A�*

A2S/average_reward_1�KA|F�+       ��K	��>�z�A�*

A2S/average_reward_1��KA\��V+       ��K	N��>�z�A�*

A2S/average_reward_1��KA�Ƚ�+       ��K	Z6�>�z�A�*

A2S/average_reward_1��LA�X[+       ��K	�a�>�z�A�*

A2S/average_reward_1��LAJ5C�+       ��K	c&?�z�A�*

A2S/average_reward_1�ROA�E++       ��K	l?�z�A�*

A2S/average_reward_1rOA��[+       ��K	_<	?�z�A�*

A2S/average_reward_18PA��+       ��K	��?�z�A�*

A2S/average_reward_1v�QA��(:+       ��K	&�?�z�A�*

A2S/average_reward_1\�RA���8+       ��K	r�?�z�A�*

A2S/average_reward_1�RA��^+       ��K	[�?�z�A�*

A2S/average_reward_1%IRA�� �+       ��K	{�?�z�A�*

A2S/average_reward_1!RA#��+       ��K	:"?�z�A�*

A2S/average_reward_1�RA>9g�+       ��K	��'?�z�A�*

A2S/average_reward_1  TAUO%�+       ��K	�&+?�z�A�*

A2S/average_reward_1��SA�e�J+       ��K	Ƴ/?�z�A�*

A2S/average_reward_1ԮTA\�r+       ��K	`�5?�z�A�*

A2S/average_reward_1.+VA�x�+       ��K	��9?�z�A�*

A2S/average_reward_1bVA�w�p+       ��K	&�<?�z�A�*

A2S/average_reward_1��UA�U�+       ��K	;~??�z�A�*

A2S/average_reward_1�UA
ߪ+       ��K	T$E?�z�A�*

A2S/average_reward_1�VA��+       ��K	�^L?�z�A�*

A2S/average_reward_1  XAy,[+       ��K	�[Q?�z�A�*

A2S/average_reward_1XA�;ut+       ��K	V?�z�A�*

A2S/average_reward_1[XAW�T�+       ��K	.�[?�z�A�*

A2S/average_reward_1��XA���+       ��K	 hb?�z�A�*

A2S/average_reward_18�YA��+       ��K	�"h?�z�A�*

A2S/average_reward_1�&ZA���t+       ��K	�Rm?�z�A�*

A2S/average_reward_1��ZA�+       ��K	�Us?�z�A�*

A2S/average_reward_1�[A��]+       ��K	bz?�z�A�*

A2S/average_reward_1�*\AN�~�+       ��K	���?�z�A�*

A2S/average_reward_1��\A/Q�J+       ��K	zh�?�z�A�*

A2S/average_reward_1��]A|{d+       ��K	ʔ�?�z�A�*

A2S/average_reward_1�_A����+       ��K	9u�?�z�A�*

A2S/average_reward_1  `A�1;8+       ��K	��?�z�A�*

A2S/average_reward_1)\_A��M+       ��K	��?�z�A�*

A2S/average_reward_1�z`ALB��+       ��K	�@�?�z�A�*

A2S/average_reward_1��`A���+       ��K	G§?�z�A�*

A2S/average_reward_1�GaA��ZU+       ��K	��?�z�A�*

A2S/average_reward_1��`Ay>�+       ��K	��?�z�A�*

A2S/average_reward_1��`A����+       ��K	W��?�z�A�*

A2S/average_reward_1��aAہv�+       ��K	�Ǻ?�z�A�*

A2S/average_reward_1{bAC�V)+       ��K	��?�z�A�*

A2S/average_reward_1�QdAt@y+       ��K	�W�?�z�A�*

A2S/average_reward_1��eA��q�+       ��K	��?�z�A�*

A2S/average_reward_1R�fA	�(+       ��K	��?�z�A�*

A2S/average_reward_133gA��:+       ��K	��?�z�A�*

A2S/average_reward_1�gA�Y�+       ��K	���?�z�A�*

A2S/average_reward_1��hA�q}[+       ��K	��?�z�A�*

A2S/average_reward_1{jAD� +       ��K	��?�z�A�*

A2S/average_reward_1q=jA<,�+       ��K	��?�z�A�*

A2S/average_reward_1��iAS�!j+       ��K	*��?�z�A�*

A2S/average_reward_1H�jA���+       ��K	���?�z�A�*

A2S/average_reward_1H�jA�P*.+       ��K	W<�?�z�A�*

A2S/average_reward_133kAv^K^�       L	vM	��"@�z�A�*�

A2S/klz)�>

A2S/average_advantage�͘>

A2S/policy_network_loss8ߘ=

A2S/value_network_losss��?

A2S/q_network_loss���?�l �+       ��K	�Q'@�z�A�*

A2S/average_reward_1
�kAc*0-+       ��K	6q,@�z�A�*

A2S/average_reward_1��lAv�(�+       ��K	��1@�z�A�*

A2S/average_reward_1��mA^�D^+       ��K	�7@�z�A�*

A2S/average_reward_1R�nA����+       ��K	�_>@�z�A�*

A2S/average_reward_1�oA�t��+       ��K	�wE@�z�A�*

A2S/average_reward_1�GqAj6f+       ��K	f�L@�z�A�*

A2S/average_reward_1{rA���+       ��K	�S@�z�A�*

A2S/average_reward_133sAU��+       ��K	�Z@�z�A�*

A2S/average_reward_1ףtAm��+       ��K	�`@�z�A�*

A2S/average_reward_1��uAB ta+       ��K	�`f@�z�A�*

A2S/average_reward_133wA�:�/+       ��K	�l@�z�A�*

A2S/average_reward_1�wAy3��+       ��K	ETt@�z�A�*

A2S/average_reward_1�pyA�L�+       ��K	��z@�z�A�*

A2S/average_reward_1H�zA�pU+       ��K	��@�z�A�*

A2S/average_reward_1R�zA��S�+       ��K	�˃@�z�A�*

A2S/average_reward_133{As�w�+       ��K	�܈@�z�A�*

A2S/average_reward_1��|A�� �+       ��K	=��@�z�A�*

A2S/average_reward_1ף|A�?�o+       ��K	���@�z�A�*

A2S/average_reward_1�}A��[+       ��K	�-�@�z�A�*

A2S/average_reward_1H�~A(K=�+       ��K	�V�@�z�A�*

A2S/average_reward_1�A�N�j+       ��K	�)�@�z�A�*

A2S/average_reward_1�A�<@)+       ��K	���@�z�A�*

A2S/average_reward_1\��A.�}7+       ��K	"��@�z�A�*

A2S/average_reward_133�A��+       ��K	<!�@�z�A�*

A2S/average_reward_1  �AS��+       ��K	�P�@�z�A�*

A2S/average_reward_1�z�AI�@+       ��K	tʳ@�z�A�*

A2S/average_reward_1H�AcD�+       ��K	5��@�z�A�*

A2S/average_reward_1H�Az�+       ��K	Ӿ�@�z�A�*

A2S/average_reward_1��A�ܘ�+       ��K	��@�z�A�*

A2S/average_reward_1�z�A �/�+       ��K	N$�@�z�A�*

A2S/average_reward_133�A0�Qv+       ��K	�@�z�A�*

A2S/average_reward_1�(�Ae�+       ��K	�)�@�z�A�*

A2S/average_reward_1�z�AG���+       ��K	�m�@�z�A�*

A2S/average_reward_1=
�Ap��[+       ��K	t�@�z�A�*

A2S/average_reward_1��A��P+       ��K	���@�z�A�*

A2S/average_reward_1���Aǰ'+       ��K	���@�z�A�*

A2S/average_reward_1q=�A��c+       ��K	m-�@�z�A�*

A2S/average_reward_1H�Aт&P+       ��K	�H�@�z�A�*

A2S/average_reward_1
׉A]��e+       ��K	�A�z�A�*

A2S/average_reward_133�A#M=}+       ��K	�_A�z�A�*

A2S/average_reward_1�p�A�?�+       ��K	A�z�A�*

A2S/average_reward_1��A����+       ��K	��A�z�A�*

A2S/average_reward_1\��A��k�+       ��K	�#A�z�A�*

A2S/average_reward_1\��AdW�+       ��K	�Y(A�z�A�*

A2S/average_reward_1ff�A�).k+       ��K	�X,A�z�A�*

A2S/average_reward_1ff�A��3�+       ��K	��2A�z�A�*

A2S/average_reward_1=
�A�%�+       ��K	�e5A�z�A�*

A2S/average_reward_1�̌A���+       ��K	�;A�z�A�*

A2S/average_reward_1=
�A�Cz+       ��K	��CA�z�A�*

A2S/average_reward_1  �A�+       ��K	��IA�z�A�*

A2S/average_reward_1��A��+       ��K	0=PA�z�A�*

A2S/average_reward_1�(�A�X2+       ��K	��TA�z�A�*

A2S/average_reward_1  �A���+       ��K	�cA�z�A�*

A2S/average_reward_1�z�A&�:�+       ��K	r�jA�z�A�*

A2S/average_reward_1=
�A����+       ��K	`[qA�z�A�*

A2S/average_reward_1�AA��+       ��K	6�uA�z�A�*

A2S/average_reward_1�AV��+       ��K	k�|A�z�A�*

A2S/average_reward_1�̒A|��+       ��K	�ͅA�z�A�*

A2S/average_reward_1\��A��r+       ��K	։A�z�A�*

A2S/average_reward_1�(�A��t�       L	vM	���A�z�A�*�

A2S/kl��=

A2S/average_advantage��>

A2S/policy_network_loss<>

A2S/value_network_loss0?^A

A2S/q_network_lossloTAcV
�+       ��K	O�A�z�A�*

A2S/average_reward_1�p�A,
��+       ��K	F�C�z�A�*

A2S/average_reward_1  �A}�e�+       ��K	d�*D�z�A�*

A2S/average_reward_1�(B��K?+       ��K	�CE�z�A�*

A2S/average_reward_1ףEB� ;�+       ��K	��aF�z�A�*

A2S/average_reward_133mBG��+       ��K	k��G�z�A�*

A2S/average_reward_1ff�B�_o�+       ��K	��H�z�A�*

A2S/average_reward_1�B���+       ��K	��I�z�A�*

A2S/average_reward_1=��B��0+       ��K	�4�J�z�A�*

A2S/average_reward_1�B�B�kfN+       ��K	��L�z�A�*

A2S/average_reward_1���B(��+       ��K	��+M�z�A�*

A2S/average_reward_1ף�B#��"+       ��K	��KN�z�A�*

A2S/average_reward_1H! C�~+       ��K	F-bO�z�A�*

A2S/average_reward_1R�	Cصl+       ��K	s�~P�z�A�*

A2S/average_reward_1\�C;�et+       ��K	���Q�z�A�*

A2S/average_reward_1ףC�H*�+       ��K	�h�R�z�A�*

A2S/average_reward_1�p'Ch7 �+       ��K	>��S�z�A�*

A2S/average_reward_1�B1Cl��+       ��K	ƍ1U�z�A�*

A2S/average_reward_1\;CX$Z�+       ��K	gcV�z�A�*

A2S/average_reward_1��DC�s�+       ��K	)��W�z�A�*

A2S/average_reward_1f�NCd2�-+       ��K	�X�z�A�*

A2S/average_reward_1��XCH�?~+       ��K	o��Y�z�A�*

A2S/average_reward_1=JbCop�+       ��K	��Z�z�A�*

A2S/average_reward_1�(lC��+       ��K	�6\�z�A�*

A2S/average_reward_1=
vC�%X�+       ��K	�]�z�A�*

A2S/average_reward_1��C��[U+       ��K	��)^�z�A�*

A2S/average_reward_1\τC�a�+       ��K	�1K_�z�A�*

A2S/average_reward_1
��C���?+       ��K	8w`�z�A�*

A2S/average_reward_1���CƒPc+       ��K	��a�z�A�*

A2S/average_reward_1���CT�f+       ��K	ڍ�b�z�A�*

A2S/average_reward_1�p�CR�Z�+       ��K	��d�z�A�*

A2S/average_reward_1)\�C�#�+       ��K	3(e�z�A�*

A2S/average_reward_1�K�CW)0+       ��K	��^f�z�A�*

A2S/average_reward_133�C>l"�+       ��K	2�sg�z�A�*

A2S/average_reward_1��Ca�Z+       ��K	7�h�z�A�*

A2S/average_reward_1f�Cq?��+       ��K	��i�z�A�*

A2S/average_reward_1
��C��(+       ��K	���j�z�A�*

A2S/average_reward_1�CX/�+       ��K	_S�k�z�A�*

A2S/average_reward_1�пC�'j+       ��K	=h*m�z�A�*

A2S/average_reward_1H��C�^�D+       ��K	��Jn�z�A�*

A2S/average_reward_1���CC��]+       ��K	Bvo�z�A�*

A2S/average_reward_1R��CT��+       ��K	��p�z�A�*

A2S/average_reward_1���C�>�
+       ��K	G��q�z�A�*

A2S/average_reward_1�l�C7'g+       ��K	no�r�z�A�*

A2S/average_reward_13S�C��L+       ��K	��t�z�A�*

A2S/average_reward_1R8�C��׃+       ��K	ˣAu�z�A�*

A2S/average_reward_1q�CK��+       ��K	�zv�z�A�*

A2S/average_reward_1H�C�}{+       ��K	��w�z�A�*

A2S/average_reward_1���C�tj2+       ��K	�˯x�z�A�*

A2S/average_reward_1���C���}+       ��K	���y�z�A�*

A2S/average_reward_1
��Cm3O�+       ��K	���z�z�A�*

A2S/average_reward_1���Cu±+       ��K	�"	|�z�A�*

A2S/average_reward_1{DD^�2+       ��K	�,}�z�A�*

A2S/average_reward_1
�D�w�+       ��K	IK~�z�A�*

A2S/average_reward_1�*D~�}W+       ��K	Q	a�z�A�*

A2S/average_reward_1\�	Dѝ�+       ��K	�^{��z�A�*

A2S/average_reward_1�D�E�+       ��K	 ᩁ�z�A�*

A2S/average_reward_1��DS��+       ��K	׮���z�A�*

A2S/average_reward_1�DJ,C+       ��K	�]҃�z�A�*

A2S/average_reward_1=zD!_n+       ��K	+~���z�A�*

A2S/average_reward_1��D�dT��       �v@�	L�2��z�A��*�

A2S/kl��7>

A2S/average_advantage}�>

A2S/policy_network_loss�lJ>

A2S/value_network_loss�7�C

A2S/q_network_loss �C��A,       ���E	�37��z�A��*

A2S/average_reward_1 �DXh,       ���E	u�X��z�A��*

A2S/average_reward_1�ED+o�,       ���E	7���z�A��*

A2S/average_reward_1��D���,       ���E	d<���z�A��*

A2S/average_reward_1�+D�!E,       ���E	�+���z�A��*

A2S/average_reward_1\�D�9��,       ���E	�C���z�A��*

A2S/average_reward_1��D��D,       ���E	Lǉ�z�A��*

A2S/average_reward_1H�D|���,       ���E	O�Ή�z�A��*

A2S/average_reward_1��D�%��,       ���E	�Bԉ�z�A��*

A2S/average_reward_1=�D�8,       ���E	P튿z�A��*

A2S/average_reward_1) DR�?n,       ���E	P���z�A��*

A2S/average_reward_1��"D��~,       ���E	O	P��z�A��*

A2S/average_reward_1�%DNSyp,       ���E	En��z�A��*

A2S/average_reward_1fv'DTR�,       ���E	��z�A��*

A2S/average_reward_1=�)D��,       ���E	&����z�A��*

A2S/average_reward_1��)D�g��,       ���E	Hmǐ�z�A��*

A2S/average_reward_1Rh,D�t�,       ���E	����z�A��*

A2S/average_reward_1f�.D���,       ���E	(���z�A��*

A2S/average_reward_1��.D��0s,       ���E	�<��z�A��*

A2S/average_reward_1�K1D&�`�,       ���E	Z�g��z�A��*

A2S/average_reward_1
�3DA��,       ���E	����z�A��*

A2S/average_reward_1�*6D���,       ���E	����z�A��*

A2S/average_reward_1�(6DI��y,       ���E	�
���z�A��*

A2S/average_reward_1)�8D�sQ�,       ���E	�ؗ�z�A��*

A2S/average_reward_1\;D�A,       ���E	��ޗ�z�A��*

A2S/average_reward_13;D����,       ���E	`���z�A��*

A2S/average_reward_1�=Dt�4,       ���E	���z�A��*

A2S/average_reward_1q�?D<��,       ���E	e�&��z�A��*

A2S/average_reward_1�@D݆,       ���E	y�G��z�A��*

A2S/average_reward_1�xBDL	d�,       ���E	�o��z�A��*

A2S/average_reward_1��DD���,       ���E	�o���z�A��*

A2S/average_reward_1q]GD�̇,       ���E	����z�A��*

A2S/average_reward_1��IDN�� ,       ���E	4<���z�A��*

A2S/average_reward_1fFLDy���,       ���E	�����z�A��*

A2S/average_reward_1�0LD'���,       ���E	���z�A��*

A2S/average_reward_1��NDX",       ���E	�:��z�A��*

A2S/average_reward_1�QD]#+,       ���E	r�:��z�A��*

A2S/average_reward_1�SDI^6�,       ���E	��A��z�A��*

A2S/average_reward_1R�SD��7�,       ���E	��|��z�A��*

A2S/average_reward_1 �UD(,2V,       ���E	]⍥�z�A��*

A2S/average_reward_1ffXD(u��,       ���E	����z�A��*

A2S/average_reward_1{�ZDP��,       ���E	�٧�z�A��*

A2S/average_reward_1{�ZD5-D,       ���E	�����z�A��*

A2S/average_reward_1{�ZDvz��,       ���E	�l:��z�A��*

A2S/average_reward_1{�ZD(�w,       ���E	�#f��z�A��*

A2S/average_reward_1{�ZD�S#u,       ���E	m`���z�A��*

A2S/average_reward_1{�ZDr��h,       ���E	��ͭ�z�A��*

A2S/average_reward_1{�ZD�,       ���E	/�宿z�A��*

A2S/average_reward_1{�ZD=�[J,       ���E	�|��z�A��*

A2S/average_reward_1{�ZD��-�,       ���E	=%��z�A��*

A2S/average_reward_1{�ZD�͕?,       ���E	w0��z�A��*

A2S/average_reward_1{�ZD�`8F,       ���E	��8��z�A��*

A2S/average_reward_1XD��.�,       ���E	B�V��z�A��*

A2S/average_reward_1XD���,       ���E	�h[��z�A��*

A2S/average_reward_1H�UD�b�,       ���E	?ei��z�A��*

A2S/average_reward_1H�UDi���,       ���E	�n��z�A��*

A2S/average_reward_1\/SDOSH�,       ���E	a]t��z�A��*

A2S/average_reward_1ͼPD}w-,       ���E	G񞵿z�A��*

A2S/average_reward_1ͼPD!/�,       ���E	�	϶�z�A��*

A2S/average_reward_1ͼPDJ�+d,       ���E	a�ֶ�z�A��*

A2S/average_reward_1�KND�����       �v@�	��÷�z�A��*�

A2S/kl{�
>

A2S/average_advantage?�=

A2S/policy_network_loss�%=

A2S/value_network_loss���C

A2S/q_network_loss�J�C���|,       ���E	��ŷ�z�A��*

A2S/average_reward_13�KD�4t,       ���E	)�ɷ�z�A��*

A2S/average_reward_1 `ID�'�,       ���E	WS˷�z�A��*

A2S/average_reward_1�FD�l�,       ���E	��ͷ�z�A��*

A2S/average_reward_1�kDD`֩�,       ���E	gӷ�z�A��*

A2S/average_reward_1��AD2},       ���E	`�շ�z�A��*

A2S/average_reward_1 �?D�O�,       ���E	��ط�z�A��*

A2S/average_reward_1�=D��gz,       ���E	�S۷�z�A��*

A2S/average_reward_1\�:D���Q,       ���E	�Y޷�z�A��*

A2S/average_reward_1
8D��@^,       ���E	cⷿz�A��*

A2S/average_reward_1 �5D�	�!,       ���E	��巿z�A��*

A2S/average_reward_1�'3D��,       ���E	+9跿z�A��*

A2S/average_reward_1q�0D93��,       ���E	�C뷿z�A��*

A2S/average_reward_1{4.D���8,       ���E	.����z�A��*

A2S/average_reward_1��+D��,       ���E	�9�z�A��*

A2S/average_reward_1HA)D���3,       ���E	\��z�A��*

A2S/average_reward_1��&D�cZ,       ���E	�����z�A��*

A2S/average_reward_1N$D6D�,       ���E	�����z�A��*

A2S/average_reward_1��!D��9,       ���E	�1���z�A��*

A2S/average_reward_13cD��&,       ���E	c ��z�A��*

A2S/average_reward_1��D?��,       ���E	�S��z�A��*

A2S/average_reward_1qmD���,       ���E	����z�A��*

A2S/average_reward_13�D̈́�d,       ���E	Qh��z�A��*

A2S/average_reward_1RxD�7�@,       ���E	Z���z�A��*

A2S/average_reward_1{D��ӯ,       ���E	ҥ��z�A��*

A2S/average_reward_1\�D���,       ���E	���z�A��*

A2S/average_reward_1�D����,       ���E	�a��z�A��*

A2S/average_reward_1��D���,       ���E	����z�A��*

A2S/average_reward_1  	D���P,       ���E	����z�A��*

A2S/average_reward_1åD��>(,       ���E	d ��z�A��*

A2S/average_reward_1�+D�m7&,       ���E	o�$��z�A��*

A2S/average_reward_1�D��,       ���E	b/+��z�A��*

A2S/average_reward_1���C �W`,       ���E	;0��z�A��*

A2S/average_reward_1���C�m,       ���E	��3��z�A��*

A2S/average_reward_1���C�˯�,       ���E	sd6��z�A��*

A2S/average_reward_13��Cx	6S,       ���E	߭:��z�A��*

A2S/average_reward_1��Cj�e,       ���E	�>��z�A��*

A2S/average_reward_13��C=�e�,       ���E	�,@��z�A��*

A2S/average_reward_1)��CD�!�,       ���E	8C��z�A��*

A2S/average_reward_1���C�B�,       ���E	�E��z�A��*

A2S/average_reward_1�9�C�N�U,       ���E	��K��z�A��*

A2S/average_reward_1�9�C��[k,       ���E	��M��z�A��*

A2S/average_reward_1E�C����,       ���E	8�P��z�A��*

A2S/average_reward_1{T�C���q,       ���E	��T��z�A��*

A2S/average_reward_1�g�CQC��,       ���E	�V��z�A��*

A2S/average_reward_1�q�C����,       ���E	��Y��z�A��*

A2S/average_reward_1e�C�_rk,       ���E	1�[��z�A��*

A2S/average_reward_1�U�C9�h,       ���E	�M^��z�A��*

A2S/average_reward_1�B�C�:`,       ���E	u�a��z�A��*

A2S/average_reward_1R8�C)��,       ���E	��d��z�A��*

A2S/average_reward_1�B�C��re,       ���E	N�g��z�A��*

A2S/average_reward_1�L�Cm��,       ���E	�j��z�A��*

A2S/average_reward_1
W�C��,       ���E	�{n��z�A��*

A2S/average_reward_1ff�C$rs,       ���E	Uq��z�A��*

A2S/average_reward_1�q�Cΰ�,       ���E	d�t��z�A��*

A2S/average_reward_1q]�C;K�,,       ���E	��w��z�A��*

A2S/average_reward_1�k�C#�:,       ���E	yz��z�A��*

A2S/average_reward_1Rx�C0/3v,       ���E	��|��z�A��*

A2S/average_reward_1�h�CX7�,       ���E	�Y��z�A��*

A2S/average_reward_1�u�C����,       ���E	����z�A��*

A2S/average_reward_1=��C{'3��       �v@�	\����z�A��*�

A2S/kl`8�>

A2S/average_advantageLs'�

A2S/policy_network_loss*2�

A2S/value_network_loss�,�?

A2S/q_network_loss>�@[δg,       ���E	`\���z�A��*

A2S/average_reward_1Õ�C<���,       ���E	�˶��z�A��*

A2S/average_reward_1���C���,       ���E	
���z�A��*

A2S/average_reward_1���C̦�,       ���E	ܔ���z�A��*

A2S/average_reward_1���C��,       ���E	�¸�z�A��*

A2S/average_reward_1��C�(?,       ���E	��Ƹ�z�A��*

A2S/average_reward_1
��C�x�,       ���E	ۭ˸�z�A��*

A2S/average_reward_1\ρCmRX�,       ���E	�Oθ�z�A��*

A2S/average_reward_1���CA�9Q,       ���E	�Ӹ�z�A��*

A2S/average_reward_1��yC��9�,       ���E	�ո�z�A��*

A2S/average_reward_1��oC��n|,       ���E	bظ�z�A��*

A2S/average_reward_1f�eC�ċ1,       ���E	�'ݸ�z�A��*

A2S/average_reward_1\\C�(�l,       ���E	E�⸿z�A��*

A2S/average_reward_1 @RCOv�,       ���E	�o渿z�A��*

A2S/average_reward_1�:RCZJ�,       ���E	,�鸿z�A��*

A2S/average_reward_1�YHC��D~,       ���E	TK븿z�A��*

A2S/average_reward_1�k>C<��,       ���E	�︿z�A��*

A2S/average_reward_1
�4Ci-�/,       ���E	��z�A��*

A2S/average_reward_1��4C<	%�,       ���E	!����z�A��*

A2S/average_reward_1��*C��u,       ���E	�+���z�A��*

A2S/average_reward_1õ CA��,       ���E	o����z�A��*

A2S/average_reward_1=�C�I,       ���E	,r��z�A��*

A2S/average_reward_13�C$~�,       ���E	0o��z�A��*

A2S/average_reward_1�C=��,       ���E	�!��z�A��*

A2S/average_reward_1��Bo^q�,       ���E	���z�A��*

A2S/average_reward_1q��B����,       ���E	���z�A��*

A2S/average_reward_1���B�\�o,       ���E	\��z�A��*

A2S/average_reward_1.�B	cG�,       ���E	��z�A��*

A2S/average_reward_1 ��B�m��,       ���E	ya��z�A��*

A2S/average_reward_1�B؂��,       ���E	�9 ��z�A��*

A2S/average_reward_1�QxB,�x�,       ���E	��#��z�A��*

A2S/average_reward_1��PBz��*,       ���E	��'��z�A��*

A2S/average_reward_1\�PB1,       ���E	l*��z�A��*

A2S/average_reward_1��(B��1 ,       ���E	`�-��z�A��*

A2S/average_reward_1��(B����,       ���E	�+3��z�A��*

A2S/average_reward_1�B%��,       ���E	�:6��z�A��*

A2S/average_reward_1{B>P�Q,       ���E	(.9��z�A��*

A2S/average_reward_1� B�B|:,       ���E	x�;��z�A��*

A2S/average_reward_1{�A��2,       ���E	?>��z�A��*

A2S/average_reward_1��EA�A0,       ���E	;KA��z�A��*

A2S/average_reward_1  DAwq/�,       ���E	�C��z�A��*

A2S/average_reward_1)\CA�Ghs,       ���E	�rG��z�A��*

A2S/average_reward_1\�BA����,       ���E	yaK��z�A��*

A2S/average_reward_1�CAׁ�,       ���E	��N��z�A��*

A2S/average_reward_1)\CA*!�,       ���E	�@Q��z�A��*

A2S/average_reward_1�AA=�,       ���E	��S��z�A��*

A2S/average_reward_1��@As�*,       ���E	/X��z�A��*

A2S/average_reward_1��@AGQ��,       ���E	�\��z�A��*

A2S/average_reward_1{BA�+��,       ���E	�Ba��z�A��*

A2S/average_reward_1=
CA�d�,       ���E	�d��z�A��*

A2S/average_reward_1R�BA<R�L,       ���E	�g��z�A��*

A2S/average_reward_1R�BA@�w,       ���E	�l��z�A��*

A2S/average_reward_1�CAK��",       ���E	�Lp��z�A��*

A2S/average_reward_1�(DA���,       ���E	�qt��z�A��*

A2S/average_reward_1��DA�k�*,       ���E	+�w��z�A��*

A2S/average_reward_1�GEA��)?,       ���E	u${��z�A��*

A2S/average_reward_1��EA�(+,       ���E	�K��z�A��*

A2S/average_reward_1ffFA�i2�,       ���E	����z�A��*

A2S/average_reward_1R�FA*���,       ���E	�"���z�A��*

A2S/average_reward_1R�FAy�vi,       ���E	��z�A��*

A2S/average_reward_1�(HA黟,       ���E	�����z�A��*

A2S/average_reward_1�zHA�=�,       ���E	�e���z�A��*

A2S/average_reward_1�pIAF���,       ���E	@Δ��z�A��*

A2S/average_reward_1��IA`7,       ���E	Jq���z�A��*

A2S/average_reward_1�zHA�]l�,       ���E	�����z�A��*

A2S/average_reward_1
�GA��<$,       ���E	N����z�A��*

A2S/average_reward_1  HA�v�,       ���E	%9���z�A��*

A2S/average_reward_1ףHAD",       ���E	�����z�A��*

A2S/average_reward_1��IA{<q�,       ���E	����z�A��*

A2S/average_reward_1q=JAq�j�,       ���E	߷���z�A��*

A2S/average_reward_1R�JAGP6,       ���E	5���z�A��*

A2S/average_reward_1\�JA�r�,       ���E	BC���z�A��*

A2S/average_reward_1ףHA�C�,       ���E	v����z�A��*

A2S/average_reward_1�IA���U,       ���E	U���z�A��*

A2S/average_reward_1ףHA�U6,       ���E	{����z�A��*

A2S/average_reward_1�pIA���,       ���E	p�Ź�z�A��*

A2S/average_reward_1��IA���},       ���E	1yʹ�z�A��*

A2S/average_reward_1{JA��,q,       ���E	A{͹�z�A��*

A2S/average_reward_1\�JA��R�,       ���E	x�ҹ�z�A��*

A2S/average_reward_1�(LAz��,       ���E	G�ֹ�z�A��*

A2S/average_reward_1ףLA�l��       �v@�	ڗ	��z�A޾*�

A2S/kl�*=

A2S/average_advantage�;�=

A2S/policy_network_loss��c<

A2S/value_network_lossx}Z@

A2S/q_network_loss�K@l˰,       ���E	�P��z�A޾*

A2S/average_reward_1��MA����,       ���E	�&��z�A޾*

A2S/average_reward_1H�ZA��},       ���E	rI��z�A޾*

A2S/average_reward_1�mA$�<,       ���E	�db��z�A޾*

A2S/average_reward_1  xA�5�u,       ���E	9j��z�A޾*

A2S/average_reward_133{Axl��,       ���E	��r��z�A޾*

A2S/average_reward_1��}A%N�,       ���E	撣��z�A޾*

A2S/average_reward_1�(�A���,       ���E	�ſ��z�A޾*

A2S/average_reward_1�z�AQ�,       ���E	o�Ǻ�z�A޾*

A2S/average_reward_1{�A�/�,       ���E	=��z�A޾*

A2S/average_reward_1�p�A��sV,       ���E	���z�A޾*

A2S/average_reward_1R��AV���,       ���E	�-��z�A޾*

A2S/average_reward_1=
�AC��,       ���E	i|B��z�A޾*

A2S/average_reward_1\��A��
',       ���E	��h��z�A޾*

A2S/average_reward_1���A��`G,       ���E	�xv��z�A޾*

A2S/average_reward_1=
�A&w�e,       ���E	�'���z�A޾*

A2S/average_reward_1  �A_��b,       ���E	}���z�A޾*

A2S/average_reward_1)\�A>�[,       ���E	����z�A޾*

A2S/average_reward_1R��A��,       ���E	`=̻�z�A޾*

A2S/average_reward_1���A��t,       ���E	%+�z�A޾*

A2S/average_reward_1���A4�V�,       ���E	sP��z�A޾*

A2S/average_reward_1��A�k��,       ���E	|�,��z�A޾*

A2S/average_reward_1ף�A�^�,       ���E	΍_��z�A޾*

A2S/average_reward_1q=�Ai�,       ���E	��k��z�A޾*

A2S/average_reward_1   B����,       ���E	`#���z�A޾*

A2S/average_reward_1ףB�^ۘ,       ���E	����z�A޾*

A2S/average_reward_1�pBW��x,       ���E	dۼ�z�A޾*

A2S/average_reward_1)\B�̋P,       ���E	&���z�A޾*

A2S/average_reward_1{BbG�;,       ���E	D���z�A޾*

A2S/average_reward_1�zBF���,       ���E	\� ��z�A޾*

A2S/average_reward_1�GB��x�,       ���E	e�A��z�A޾*

A2S/average_reward_1�B�maK,       ���E	x�I��z�A޾*

A2S/average_reward_1q=Byon,       ���E	�b��z�A޾*

A2S/average_reward_133B
Ɓ~,       ���E	Łf��z�A޾*

A2S/average_reward_1{B5߽�,       ���E	&Dq��z�A޾*

A2S/average_reward_1H�BYӦ�,       ���E	�Y���z�A޾*

A2S/average_reward_1  BK(�i,       ���E	�̬��z�A޾*

A2S/average_reward_1��#B�[�0,       ���E	,H⽿z�A޾*

A2S/average_reward_1ff*BFq<+,       ���E	�5���z�A޾*

A2S/average_reward_1�p.B*b�,       ���E	�#%��z�A޾*

A2S/average_reward_1ff4B�0s�,       ���E	U�A��z�A޾*

A2S/average_reward_1R�8Bs><k,       ���E	�c��z�A޾*

A2S/average_reward_1�Q=B���~,       ���E	b����z�A޾*

A2S/average_reward_1��BB�,       ���E	#g���z�A޾*

A2S/average_reward_1R�EBH�)Q,       ���E	:}���z�A޾*

A2S/average_reward_1�(GB��Q>,       ���E	�eԾ�z�A޾*

A2S/average_reward_1  LB��J�,       ���E	��z�A޾*

A2S/average_reward_1��OB���,       ���E	i]���z�A޾*

A2S/average_reward_1�PB���Y,       ���E	�^%��z�A޾*

A2S/average_reward_1H�UB�J�e,       ���E	�L��z�A޾*

A2S/average_reward_1�pZB���,       ���E	�_l��z�A޾*

A2S/average_reward_1�G^B�Q�m,       ���E	s|���z�A޾*

A2S/average_reward_1
�cB�GV�,       ���E	�𥿿z�A޾*

A2S/average_reward_1R�dB�GX],       ���E	�l���z�A޾*

A2S/average_reward_1
�eB��,       ���E	n,���z�A޾*

A2S/average_reward_1�fBI�,       ���E		쿿z�A޾*

A2S/average_reward_1�(kB�=�,       ���E	t��z�A޾*

A2S/average_reward_1�znB�p�,       ���E	�i��z�A޾*

A2S/average_reward_1
�oBW���,       ���E	x�:��z�A޾*

A2S/average_reward_1�ztB�,},       ���E	�Z��z�A޾*

A2S/average_reward_1�QxB���u,       ���E	W����z�A޾*

A2S/average_reward_1��}B���,       ���E	|���z�A޾*

A2S/average_reward_1�G�B�8��,       ���E	L����z�A޾*

A2S/average_reward_133�B&!=,       ���E	�R��z�A޾*

A2S/average_reward_1�L�Bg
�,       ���E	B�$��z�A޾*

A2S/average_reward_1��B��,       ���E	��R��z�A޾*

A2S/average_reward_1�ыB�*y,       ���E	ʂ��z�A޾*

A2S/average_reward_1�̎B��2.,       ���E	�v���z�A޾*

A2S/average_reward_1  �BE64K,       ���E	>9���z�A޾*

A2S/average_reward_1\��B�Z�,       ���E	�����z�A޾*

A2S/average_reward_1H�B��l�,       ���E	����z�A޾*

A2S/average_reward_1�ǔB	tfI,       ���E	��¿z�A޾*

A2S/average_reward_1�B��,       ���E	��¿z�A޾*

A2S/average_reward_1H�B�V=,       ���E	7�%¿z�A޾*

A2S/average_reward_1R8�B~Ao,       ���E	 3¿z�A޾*

A2S/average_reward_13��BPG,       ���E	�@¿z�A޾*

A2S/average_reward_1�Q�B�)�,       ���E	�Q_¿z�A޾*

A2S/average_reward_1  �B�,       ���E	b�{¿z�A޾*

A2S/average_reward_1\��B�R�Y,       ���E	�^�¿z�A޾*

A2S/average_reward_1�B�Bɇ,       ���E	XT�¿z�A޾*

A2S/average_reward_1.�B�c˵,       ���E	�&�¿z�A޾*

A2S/average_reward_1Ha�B��v,       ���E	��¿z�A޾*

A2S/average_reward_1���BÞp�,       ���E	��ÿz�A޾*

A2S/average_reward_1�ѥB�v�b,       ���E	@q5ÿz�A޾*

A2S/average_reward_1ף�B�GB[,       ���E	Q�Jÿz�A޾*

A2S/average_reward_1�B'��,       ���E	[�Sÿz�A޾*

A2S/average_reward_1 ��BN��d,       ���E	�Fsÿz�A޾*

A2S/average_reward_1���B��,       ���E	�9zÿz�A޾*

A2S/average_reward_1
׫B|l��,       ���E	m��ÿz�A޾*

A2S/average_reward_1�B�B�mK,       ���E	ج�ÿz�A޾*

A2S/average_reward_1)ܰB����,       ���E	@�ÿz�A޾*

A2S/average_reward_1�z�B�ܟ�,       ���E	�0Ŀz�A޾*

A2S/average_reward_1��B��W+,       ���E	��Ŀz�A޾*

A2S/average_reward_133�B�3y�,       ���E	'*=Ŀz�A޾*

A2S/average_reward_1�¸B}���,       ���E	��cĿz�A޾*

A2S/average_reward_1Ha�B,1�#,       ���E	�o�Ŀz�A޾*

A2S/average_reward_1 ��Brx��,       ���E	eA�Ŀz�A޾*

A2S/average_reward_1�p�Bpeb�,       ���E	��Ŀz�A޾*

A2S/average_reward_13��B��M,       ���E	`
�Ŀz�A޾*

A2S/average_reward_1���B눑z,       ���E	qſz�A޾*

A2S/average_reward_1=��B���       �v@�	�Uſz�A��*�

A2S/kl�+>

A2S/average_advantage�I�=

A2S/policy_network_loss��>

A2S/value_network_lossx+C

A2S/q_network_loss8�Cb�2A,       ���E	#/Yſz�A��*

A2S/average_reward_1ff�B'�S,       ���E	#�_ſz�A��*

A2S/average_reward_1��Bʔ�I,       ���E	��cſz�A��*

A2S/average_reward_1)��B�DiU,       ���E	>�hſz�A��*

A2S/average_reward_1��B@W�,       ���E	�Sqſz�A��*

A2S/average_reward_1�z�B16�m,       ���E	�tzſz�A��*

A2S/average_reward_1�p�Bg/b�,       ���E	�J�ſz�A��*

A2S/average_reward_1ff�Ba��J,       ���E	<Άſz�A��*

A2S/average_reward_1{��BU��H,       ���E	���ſz�A��*

A2S/average_reward_1�Q�B+%-�,       ���E	!m�ſz�A��*

A2S/average_reward_1���BK���,       ���E	ݖſz�A��*

A2S/average_reward_1���B��^�,       ���E	��ſz�A��*

A2S/average_reward_1��B	fO,       ���E	.��ſz�A��*

A2S/average_reward_1)ܴB���,       ���E	5�ſz�A��*

A2S/average_reward_1�z�Bҹ9V,       ���E	O­ſz�A��*

A2S/average_reward_133�Bt�
3,       ���E	e��ſz�A��*

A2S/average_reward_1R8�B���,       ���E	�K�ſz�A��*

A2S/average_reward_1�Bb,       ���E	��ſz�A��*

A2S/average_reward_1f�B��V0,       ���E	x�ſz�A��*

A2S/average_reward_1)ܭB��p,       ���E	�U�ſz�A��*

A2S/average_reward_1���B9�,       ���E	��ſz�A��*

A2S/average_reward_1=��B9��<,       ���E	���ſz�A��*

A2S/average_reward_1{�B��,       ���E	�X�ſz�A��*

A2S/average_reward_1.�BY���,       ���E	���ſz�A��*

A2S/average_reward_1H�B�߹z,       ���E	��ſz�A��*

A2S/average_reward_1�Q�BU��,       ���E	��ſz�A��*

A2S/average_reward_1��B�f�,       ���E	�E�ſz�A��*

A2S/average_reward_1 ��B�dM�,       ���E	{�ſz�A��*

A2S/average_reward_1ff�B�(��,       ���E	�ƿz�A��*

A2S/average_reward_1�p�B��[�,       ���E	�Fƿz�A��*

A2S/average_reward_1.�BQK�,       ���E	�Lƿz�A��*

A2S/average_reward_1��B+}�u,       ���E	�ƿz�A��*

A2S/average_reward_1�B�U�,       ���E	��ƿz�A��*

A2S/average_reward_1�k�B*f�,       ���E	m#ƿz�A��*

A2S/average_reward_1�B�R��,       ���E	�(ƿz�A��*

A2S/average_reward_1�k�B"PN-,       ���E	`�,ƿz�A��*

A2S/average_reward_1=
�B�,`�,       ���E	޼2ƿz�A��*

A2S/average_reward_1��B���,       ���E	d�=ƿz�A��*

A2S/average_reward_1�̑B�N;�,       ���E	��Eƿz�A��*

A2S/average_reward_1�(�B���,       ���E	�%Mƿz�A��*

A2S/average_reward_1��B��2,       ���E	�Sƿz�A��*

A2S/average_reward_1{��B=�,       ���E	��`ƿz�A��*

A2S/average_reward_1��Bߺ��,       ���E	A�gƿz�A��*

A2S/average_reward_1��BY��,       ���E	��lƿz�A��*

A2S/average_reward_13��B߼��,       ���E	��uƿz�A��*

A2S/average_reward_1Ha�BG
�,       ���E	[{ƿz�A��*

A2S/average_reward_1�(�B�:�,       ���E	��ƿz�A��*

A2S/average_reward_1���B�0<�,       ���E	�[�ƿz�A��*

A2S/average_reward_1��B���,       ���E	��ƿz�A��*

A2S/average_reward_1R�}B��!,       ���E	��ƿz�A��*

A2S/average_reward_1�yB����,       ���E	y��ƿz�A��*

A2S/average_reward_1ףuB�`�K,       ���E	�a�ƿz�A��*

A2S/average_reward_1�ppB�ǌT,       ���E	*N�ƿz�A��*

A2S/average_reward_1�pB�S�2,       ���E	kܱƿz�A��*

A2S/average_reward_1{pB['�,       ���E	#�ƿz�A��*

A2S/average_reward_1�ppB����,       ���E	���ƿz�A��*

A2S/average_reward_1H�kBɜO�,       ���E	��ƿz�A��*

A2S/average_reward_1��hB���,       ���E	J$�ƿz�A��*

A2S/average_reward_1��gB��,       ���E	���ƿz�A��*

A2S/average_reward_1��cB�s,       ���E	Ϳ�ƿz�A��*

A2S/average_reward_1�z`B*��,       ���E	���ƿz�A��*

A2S/average_reward_1��[B]��,       ���E	�Z�ƿz�A��*

A2S/average_reward_1�WB.��,       ���E	d��ƿz�A��*

A2S/average_reward_1ffQB)��[,       ���E	*��ƿz�A��*

A2S/average_reward_1�pKB2RR�,       ���E	���ƿz�A��*

A2S/average_reward_133IB���-,       ���E	G�ǿz�A��*

A2S/average_reward_1�GDB��U�,       ���E	K6ǿz�A��*

A2S/average_reward_1��>B���,       ���E	:�ǿz�A��*

A2S/average_reward_1�:B�/ڤ,       ���E	i�ǿz�A��*

A2S/average_reward_1�p7B��,       ���E	I�$ǿz�A��*

A2S/average_reward_1�z7BT��,       ���E	��)ǿz�A��*

A2S/average_reward_1��3B ��q,       ���E	;F4ǿz�A��*

A2S/average_reward_1�z0B�R,�,       ���E	O�:ǿz�A��*

A2S/average_reward_1\�.B�R8�,       ���E	QDǿz�A��*

A2S/average_reward_1�z.B����,       ���E	һMǿz�A��*

A2S/average_reward_1)\.B��Q�,       ���E	\Vǿz�A��*

A2S/average_reward_1H�-B-~�,       ���E	�_ǿz�A��*

A2S/average_reward_1ff+Bؘ�,       ���E	��cǿz�A��*

A2S/average_reward_1)\(B�UǊ,       ���E	��hǿz�A��*

A2S/average_reward_1H�$B��,       ���E	I4pǿz�A��*

A2S/average_reward_1\�B�T	�,       ���E	��yǿz�A��*

A2S/average_reward_1  B�z},       ���E	w�~ǿz�A��*

A2S/average_reward_1ףBӚ�,       ���E	���ǿz�A��*

A2S/average_reward_1  B���[,       ���E	�֏ǿz�A��*

A2S/average_reward_1�(B-w,       ���E	୘ǿz�A��*

A2S/average_reward_1q=B�R|,       ���E	뱝ǿz�A��*

A2S/average_reward_1\�BPZE(,       ���E	 ��ǿz�A��*

A2S/average_reward_1ffB{��=,       ���E	_X�ǿz�A��*

A2S/average_reward_1  B�k�,       ���E	H�ǿz�A��*

A2S/average_reward_1�	B��,       ���E	�j�ǿz�A��*

A2S/average_reward_1\�B'Q@�,       ���E	Fl�ǿz�A��*

A2S/average_reward_1=
�A��#,       ���E	km�ǿz�A��*

A2S/average_reward_1�Q�A:���,       ���E	���ǿz�A��*

A2S/average_reward_1  �A0߅�,       ���E	l�ǿz�A��*

A2S/average_reward_1���AY���,       ���E	c��ǿz�A��*

A2S/average_reward_1��Al�(�,       ���E	��ǿz�A��*

A2S/average_reward_1)\�A�e�?,       ���E	=�ǿz�A��*

A2S/average_reward_1q=�A�6b
,       ���E	���ǿz�A��*

A2S/average_reward_1��A9x�,       ���E	�G�ǿz�A��*

A2S/average_reward_1�p�AqE�,       ���E	i��ǿz�A��*

A2S/average_reward_1���A��k,       ���E	�R�ǿz�A��*

A2S/average_reward_1���A,�A�,       ���E	�Xȿz�A��*

A2S/average_reward_1�̸Ar��,       ���E	�	ȿz�A��*

A2S/average_reward_1��ADI�,       ���E	Zȿz�A��*

A2S/average_reward_1�z�Aف�_,       ���E	0�ȿz�A��*

A2S/average_reward_1{�Au�+z,       ���E	��ȿz�A��*

A2S/average_reward_1���AuF�c,       ���E	��ȿz�A��*

A2S/average_reward_1���A�D*X,       ���E	�%ȿz�A��*

A2S/average_reward_1���A��h,       ���E	v�-ȿz�A��*

A2S/average_reward_1�Q�At�K.,       ���E	2ȿz�A��*

A2S/average_reward_1ff�A6Da,       ���E	K49ȿz�A��*

A2S/average_reward_1�̺Aa�ț,       ���E	��>ȿz�A��*

A2S/average_reward_1�̺A�ļ%,       ���E	��Cȿz�A��*

A2S/average_reward_1R��A����,       ���E	c�Lȿz�A��*

A2S/average_reward_133�AP�|A,       ���E	{�Vȿz�A��*

A2S/average_reward_1�»A0l�d,       ���E	�J\ȿz�A��*

A2S/average_reward_1�G�A��|�,       ���E	x�`ȿz�A��*

A2S/average_reward_1=
�A��3�,       ���E	2�kȿz�A��*

A2S/average_reward_1{�Ar��,       ���E	 �qȿz�A��*

A2S/average_reward_1
׻AY{�,       ���E	�vȿz�A��*

A2S/average_reward_1H�A�X8>�       �v@�	���ȿz�A�*�

A2S/kl�Р<

A2S/average_advantage���<

A2S/policy_network_loss���

A2S/value_network_lossw�@

A2S/q_network_loss�^�@����,       ���E	��ȿz�A�*

A2S/average_reward_1)\�A���,       ���E	y�ɿz�A�*

A2S/average_reward_1���A�
G�,       ���E	c\&ɿz�A�*

A2S/average_reward_1���A�r,       ���E	��@ɿz�A�*

A2S/average_reward_1R��A��J�,       ���E	ASaɿz�A�*

A2S/average_reward_1���A��q;,       ���E	���ɿz�A�*

A2S/average_reward_133�A����,       ���E	�$�ɿz�A�*

A2S/average_reward_1R��A�L@W,       ���E	���ɿz�A�*

A2S/average_reward_133�A����,       ���E	��ɿz�A�*

A2S/average_reward_1���A�] �,       ���E	_ʿz�A�*

A2S/average_reward_1)\B���Z,       ���E	/3$ʿz�A�*

A2S/average_reward_1�B��,       ���E	15Iʿz�A�*

A2S/average_reward_133B����,       ���E	�oʿz�A�*

A2S/average_reward_1��BM�,       ���E	���ʿz�A�*

A2S/average_reward_1��B�7B(,       ���E	D.�ʿz�A�*

A2S/average_reward_1��B�q�9,       ���E	�@�ʿz�A�*

A2S/average_reward_1�zB0�r3,       ���E	���ʿz�A�*

A2S/average_reward_1��B�H',       ���E	�
˿z�A�*

A2S/average_reward_1�QBmYJ,       ���E	ED6˿z�A�*

A2S/average_reward_1R�"B�p,       ���E	}b˿z�A�*

A2S/average_reward_1�'B_�|�,       ���E	�˿z�A�*

A2S/average_reward_1ף*B�$,       ���E	Xթ˿z�A�*

A2S/average_reward_1�z-B���,       ���E	Dv�˿z�A�*

A2S/average_reward_1ף0BF�r,       ���E	aF�˿z�A�*

A2S/average_reward_1�z2B7Q��,       ���E	��̿z�A�*

A2S/average_reward_1R�6B}���,       ���E	!+̿z�A�*

A2S/average_reward_1)\;B�I�],       ���E	U�H̿z�A�*

A2S/average_reward_1)\>B�h��,       ���E	��l̿z�A�*

A2S/average_reward_133CB /��,       ���E	!�̿z�A�*

A2S/average_reward_1q=EB���U,       ���E	���̿z�A�*

A2S/average_reward_1�pHB��?,       ���E	2)�̿z�A�*

A2S/average_reward_1�(MB��,       ���E	���̿z�A�*

A2S/average_reward_1q=PB�w�,       ���E	�hͿz�A�*

A2S/average_reward_1R�TB�k�,       ���E	�6Ϳz�A�*

A2S/average_reward_1�QWB���,       ���E	�*XͿz�A�*

A2S/average_reward_1{ZBLj�~,       ���E	�pwͿz�A�*

A2S/average_reward_1�Q]B�A�/,       ���E	mܣͿz�A�*

A2S/average_reward_1  cBh���,       ���E	9��Ϳz�A�*

A2S/average_reward_1)\gBe�*G,       ���E	N��Ϳz�A�*

A2S/average_reward_1�iB�#,       ���E	c\�Ϳz�A�*

A2S/average_reward_1  lB���,       ���E	��οz�A�*

A2S/average_reward_1��mB����,       ���E	�Tοz�A�*

A2S/average_reward_1�pB��}�,       ���E	��?οz�A�*

A2S/average_reward_1H�tB�D��,       ���E		�Qοz�A�*

A2S/average_reward_1��vB^5��,       ���E	��kοz�A�*

A2S/average_reward_1�QyB/}��,       ���E	���οz�A�*

A2S/average_reward_1R�{B�0&,       ���E	6W�οz�A�*

A2S/average_reward_1R�B�S�,       ���E	�οz�A�*

A2S/average_reward_1�сB�7��,       ���E	%b�οz�A�*

A2S/average_reward_1���B,��G,       ���E	p.�οz�A�*

A2S/average_reward_1�B�B���,       ���E	k�Ͽz�A�*

A2S/average_reward_1H�B?��,       ���E	��9Ͽz�A�*

A2S/average_reward_1���BZH,       ���E	{lXϿz�A�*

A2S/average_reward_1�Q�B�n�
,       ���E	��oϿz�A�*

A2S/average_reward_1ff�B�Cp,       ���E	���Ͽz�A�*

A2S/average_reward_1=��Bƥ��,       ���E	n�Ͽz�A�*

A2S/average_reward_1�k�B^�[p,       ���E	���Ͽz�A�*

A2S/average_reward_1�p�BK��o,       ���E	���Ͽz�A�*

A2S/average_reward_1��B9�6�,       ���E	�_пz�A�*

A2S/average_reward_1=
�B3���,       ���E	oD'пz�A�*

A2S/average_reward_1�k�B��$�,       ���E	��Bпz�A�*

A2S/average_reward_1ף�B��%},       ���E	��lпz�A�*

A2S/average_reward_1���B���",       ���E	Z��пz�A�*

A2S/average_reward_1{��B�2�,       ���E	DЯпz�A�*

A2S/average_reward_133�B{�S,       ���E	:��пz�A�*

A2S/average_reward_1���BK�BA,       ���E	,�пz�A�*

A2S/average_reward_1�ѝB҂�,       ���E	�ѿz�A�*

A2S/average_reward_1�z�B�|9�,       ���E	��:ѿz�A�*

A2S/average_reward_1
W�B!j_�,       ���E	$�`ѿz�A�*

A2S/average_reward_1�#�B���,       ���E	�G�ѿz�A�*

A2S/average_reward_1���BE:�,       ���E	�9�ѿz�A�*

A2S/average_reward_1�(�Blo!,       ���E	E�ѿz�A�*

A2S/average_reward_1Ha�B*�d�,       ���E	;��ѿz�A�*

A2S/average_reward_1R8�B?��,       ���E	��ҿz�A�*

A2S/average_reward_1.�BXƴF,       ���E	+=9ҿz�A�*

A2S/average_reward_1�(�B����,       ���E	JAdҿz�A�*

A2S/average_reward_133�B�9,       ���E	�h�ҿz�A�*

A2S/average_reward_1H�B���,       ���E	���ҿz�A�*

A2S/average_reward_1�u�B���",       ���E	�ҿz�A�*

A2S/average_reward_1�B�Bͣ9,       ���E	�p�ҿz�A�*

A2S/average_reward_1�ѶB�_�,       ���E	&y�ҿz�A�*

A2S/average_reward_1 ��B΋f�,       ���E	�ӿz�A�*

A2S/average_reward_1  �BS\/m,       ���E	�+ӿz�A�*

A2S/average_reward_1�p�Bb��,       ���E	_�4ӿz�A�*

A2S/average_reward_1��B�� �,       ���E	Z�Sӿz�A�*

A2S/average_reward_1R��B��S<,       ���E	�1vӿz�A�*

A2S/average_reward_1.�B=���,       ���E	Қӿz�A�*

A2S/average_reward_1�G�B]��",       ���E	}��ӿz�A�*

A2S/average_reward_1ff�BUb� ,       ���E	'��ӿz�A�*

A2S/average_reward_1\�Bl���,       ���E	���ӿz�A�*

A2S/average_reward_1���B6�,       ���E	;CԿz�A�*

A2S/average_reward_1�B͇�x,       ���E	�bBԿz�A�*

A2S/average_reward_1���B�v�),       ���E	�Y_Կz�A�*

A2S/average_reward_1 ��B�Sk�,       ���E	JvԿz�A�*

A2S/average_reward_1 ��BV���,       ���E	:��Կz�A�*

A2S/average_reward_1)��BO;!T,       ���E	\��Կz�A�*

A2S/average_reward_1�Q�Bi˼>,       ���E	͝�Կz�A�*

A2S/average_reward_1H��Bs�Ń,       ���E		��Կz�A�*

A2S/average_reward_1{�B"$<&,       ���E	��տz�A�*

A2S/average_reward_1�z�B	N(?,       ���E	�$(տz�A�*

A2S/average_reward_1
W�B����,       ���E	,}Lտz�A�*

A2S/average_reward_1.�B��\,       ���E	"�pտz�A�*

A2S/average_reward_1�B�B	}��,       ���E	�#�տz�A�*

A2S/average_reward_1q=�BAC��,       ���E	6�տz�A�*

A2S/average_reward_1R��Byd�h,       ���E	�l�տz�A�*

A2S/average_reward_1\�BE)q,       ���E	X��տz�A�*

A2S/average_reward_1�#�B{$��,       ���E	~�ֿz�A�*

A2S/average_reward_1�L�B|��,       ���E	�),ֿz�A�*

A2S/average_reward_1���B~��,       ���E	^k_ֿz�A�*

A2S/average_reward_1���B��2�,       ���E	�
rֿz�A�*

A2S/average_reward_1�Q�B�h3�,       ���E	? �ֿz�A�*

A2S/average_reward_1�L�BH���,       ���E	$A�ֿz�A�*

A2S/average_reward_1�BSf,       ���E	�t�ֿz�A�*

A2S/average_reward_1���BX0�,       ���E	ә�ֿz�A�*

A2S/average_reward_1�B�Ba�,       ���E	�+׿z�A�*

A2S/average_reward_1\��B���,       ���E	2)׿z�A�*

A2S/average_reward_1{�B^[{,       ���E	4KQ׿z�A�*

A2S/average_reward_1{��B��A,       ���E	�(y׿z�A�*

A2S/average_reward_133�BH ,       ���E		��׿z�A�*

A2S/average_reward_1��B��a�,       ���E	���׿z�A�*

A2S/average_reward_1�L�Bm���,       ���E	��׿z�A�*

A2S/average_reward_1��BT��,       ���E	�1ؿz�A�*

A2S/average_reward_1\��Bv��},       ���E	��ؿz�A�*

A2S/average_reward_1)��B�i;,       ���E	��Cؿz�A�*

A2S/average_reward_1{��BR?��,       ���E	M�^ؿz�A�*

A2S/average_reward_1=
�Bn���,       ���E	��wؿz�A�*

A2S/average_reward_1�Q�B��F�,       ���E	,e�ؿz�A�*

A2S/average_reward_1 ��BT��,       ���E	��ؿz�A�*

A2S/average_reward_1�u�B��6�,       ���E	���ؿz�A�*

A2S/average_reward_1
��B9�f�,       ���E	`��ؿz�A�*

A2S/average_reward_1���BY��,       ���E	^ٿz�A�*

A2S/average_reward_1ף�B�3,       ���E	�L:ٿz�A�*

A2S/average_reward_1�L�B1|K,       ���E	�+iٿz�A�*

A2S/average_reward_133�B%��,       ���E	�w�ٿz�A�*

A2S/average_reward_1�p�B��p�,       ���E	L[�ٿz�A�*

A2S/average_reward_1)\�B��*},       ���E	�ٿz�A�*

A2S/average_reward_1)\�B�s�e,       ���E	c��ٿz�A�*

A2S/average_reward_1���B�)�,       ���E	� ڿz�A�*

A2S/average_reward_1�z�B3s+E,       ���E	��%ڿz�A�*

A2S/average_reward_1���B^|�,       ���E	RUKڿz�A�*

A2S/average_reward_1\��BA�(��       �v@�	���ڿz�Aԙ*�

A2S/kl�W�=

A2S/average_advantage)k�=

A2S/policy_network_loss\!E�

A2S/value_network_lossG
3B

A2S/q_network_loss��AO=w9,       ���E	���ڿz�Aԙ*

A2S/average_reward_1=��B��,       ���E	;��ڿz�Aԙ*

A2S/average_reward_1H��Bk�,       ���E	��ۿz�Aԙ*

A2S/average_reward_1\�Bг��,       ���E	8
,ۿz�Aԙ*

A2S/average_reward_133�BQD(,       ���E	͒Nۿz�Aԙ*

A2S/average_reward_1��B�%O,       ���E	�Nvۿz�Aԙ*

A2S/average_reward_1)\�B��,       ���E	+˟ۿz�Aԙ*

A2S/average_reward_1��B2b F,       ���E	+��ۿz�Aԙ*

A2S/average_reward_1�u�Bv�e,       ���E	���ۿz�Aԙ*

A2S/average_reward_1\��B����,       ���E	�K�ۿz�Aԙ*

A2S/average_reward_1
��Bu�",       ���E	]�ܿz�Aԙ*

A2S/average_reward_133�B��ݻ,       ���E	zAܿz�Aԙ*

A2S/average_reward_1��B�A�?,       ���E	|�dܿz�Aԙ*

A2S/average_reward_1{��Bj�[w,       ���E	���ܿz�Aԙ*

A2S/average_reward_1R��B]�V�,       ���E	�S�ܿz�Aԙ*

A2S/average_reward_1\��B���v,       ���E	��ܿz�Aԙ*

A2S/average_reward_1�p�B�0�e,       ���E	8/�ܿz�Aԙ*

A2S/average_reward_1=
�B����,       ���E	/�ݿz�Aԙ*

A2S/average_reward_1f��B��x;,       ���E	�RBݿz�Aԙ*

A2S/average_reward_1���B3c�,       ���E	��aݿz�Aԙ*

A2S/average_reward_1��B0N�,       ���E	욀ݿz�Aԙ*

A2S/average_reward_1���B�ss,       ���E	��ݿz�Aԙ*

A2S/average_reward_1R��B���,       ���E	�i�ݿz�Aԙ*

A2S/average_reward_1f��BP:|�,       ���E	�	�ݿz�Aԙ*

A2S/average_reward_1�Q�B��x,       ���E	��!޿z�Aԙ*

A2S/average_reward_1�#�B/t@,       ���E	�*J޿z�Aԙ*

A2S/average_reward_1�G�B���,       ���E	z�n޿z�Aԙ*

A2S/average_reward_1���BOw(�,       ���E	5��޿z�Aԙ*

A2S/average_reward_1=��BLO�8,       ���E	 �޿z�Aԙ*

A2S/average_reward_1{��B��q,       ���E	���޿z�Aԙ*

A2S/average_reward_1{�B� �,       ���E	I��޿z�Aԙ*

A2S/average_reward_1���B�e� ,       ���E	�/ ߿z�Aԙ*

A2S/average_reward_1���B���,       ���E	�cM߿z�Aԙ*

A2S/average_reward_1�z�B#V,       ���E	�r߿z�Aԙ*

A2S/average_reward_1��B���X,       ���E	���߿z�Aԙ*

A2S/average_reward_1���B��,       ���E	{��߿z�Aԙ*

A2S/average_reward_1���B"�N�,       ���E	��߿z�Aԙ*

A2S/average_reward_1�(�B��il,       ���E	.�߿z�Aԙ*

A2S/average_reward_1=��B.�+,       ���E	�{�z�Aԙ*

A2S/average_reward_1Ha�B��`O,       ���E	�D�z�Aԙ*

A2S/average_reward_1�#�B+��p,       ���E	�ud�z�Aԙ*

A2S/average_reward_1ff�Bd�,       ���E	g8��z�Aԙ*

A2S/average_reward_1��B��d,       ���E	�\��z�Aԙ*

A2S/average_reward_1���B�fZ^,       ���E	a
��z�Aԙ*

A2S/average_reward_1=��B�0�t,       ���E	�s�z�Aԙ*

A2S/average_reward_13��BI�Ш,       ���E	��"�z�Aԙ*

A2S/average_reward_1\�B�JS�,       ���E	�t<�z�Aԙ*

A2S/average_reward_1\��Bv),       ���E	�JY�z�Aԙ*

A2S/average_reward_1=
�B3j^,       ���E	s�y�z�Aԙ*

A2S/average_reward_133�B+�gG,       ���E	���z�Aԙ*

A2S/average_reward_1=��B$�,       ���E	����z�Aԙ*

A2S/average_reward_1�k�B[(!�,       ���E	����z�Aԙ*

A2S/average_reward_1)\�BY�,       ���E	*�	�z�Aԙ*

A2S/average_reward_1ף�B��:�,       ���E	�,�z�Aԙ*

A2S/average_reward_1�k�B��Ɉ,       ���E	�I�z�Aԙ*

A2S/average_reward_1 ��B���m,       ���E	��n�z�Aԙ*

A2S/average_reward_1��B78n�,       ���E	B��z�Aԙ*

A2S/average_reward_1�Q�B�c�,       ���E	����z�Aԙ*

A2S/average_reward_1{�BO��,       ���E	 ���z�Aԙ*

A2S/average_reward_1���B�-�,       ���E	�s �z�Aԙ*

A2S/average_reward_1�G�B��=�,       ���E	в�z�Aԙ*

A2S/average_reward_1���B;#��,       ���E	wpJ�z�Aԙ*

A2S/average_reward_1���BT^��,       ���E	Dfw�z�Aԙ*

A2S/average_reward_1{��BsZ��,       ���E	ˤ�z�Aԙ*

A2S/average_reward_1�Q�B�P��,       ���E	����z�Aԙ*

A2S/average_reward_1q��B��
�,       ���E	����z�Aԙ*

A2S/average_reward_1{�B��8,       ���E	�t�z�Aԙ*

A2S/average_reward_133�B�?[,       ���E	�9�z�Aԙ*

A2S/average_reward_1f��B��+�,       ���E	8a�z�Aԙ*

A2S/average_reward_1=��B���r,       ���E	��z�Aԙ*

A2S/average_reward_1ף�B��9|,       ���E	2Ƿ�z�Aԙ*

A2S/average_reward_1���BH�'�,       ���E	�F��z�Aԙ*

A2S/average_reward_1q=�B
�H,       ���E	�~	�z�Aԙ*

A2S/average_reward_1=��B:���,       ���E	�%*�z�Aԙ*

A2S/average_reward_1�u�BXQ��,       ���E	�L�z�Aԙ*

A2S/average_reward_1���B��~u,       ���E	np�z�Aԙ*

A2S/average_reward_1���B����,       ���E	]���z�Aԙ*

A2S/average_reward_1=��BBp��,       ���E	�O��z�Aԙ*

A2S/average_reward_1{��B��\b,       ���E	����z�Aԙ*

A2S/average_reward_1)\�B�h�,       ���E	�:��z�Aԙ*

A2S/average_reward_1�L�B��,       ���E	��z�Aԙ*

A2S/average_reward_1���B'�Y�,       ���E	��2�z�Aԙ*

A2S/average_reward_1���B�kc�,       ���E	�T�z�Aԙ*

A2S/average_reward_1.�B�y��,       ���E	ρt�z�Aԙ*

A2S/average_reward_1�Q�B3�F�,       ���E	ʐ�z�Aԙ*

A2S/average_reward_1�B�B���g,       ���E	���z�Aԙ*

A2S/average_reward_1  �B�ˍI,       ���E	F���z�Aԙ*

A2S/average_reward_1�G�Bʧ�V,       ���E		�z�Aԙ*

A2S/average_reward_1)\�B~#�O,       ���E	F�1�z�Aԙ*

A2S/average_reward_1)��B[��},       ���E	6dY�z�Aԙ*

A2S/average_reward_1ף�B���L,       ���E	���z�Aԙ*

A2S/average_reward_1)��B�JT�,       ���E	2���z�Aԙ*

A2S/average_reward_13��B�6k,       ���E	Aj��z�Aԙ*

A2S/average_reward_1R��Bp׆�,       ���E	���z�Aԙ*

A2S/average_reward_1.�B=��,       ���E	� �z�Aԙ*

A2S/average_reward_1{�B��|�,       ���E	�'C�z�Aԙ*

A2S/average_reward_1=��BǞ�/,       ���E	��h�z�Aԙ*

A2S/average_reward_13��B:mJ�,       ���E	ڐ�z�Aԙ*

A2S/average_reward_1���B��O,       ���E	���z�Aԙ*

A2S/average_reward_1���Bà�,       ���E	$���z�Aԙ*

A2S/average_reward_1)\�B��\�,       ���E	���z�Aԙ*

A2S/average_reward_1���B��m,       ���E	�Y�z�Aԙ*

A2S/average_reward_1��B�23,       ���E	�Y8�z�Aԙ*

A2S/average_reward_1{��Bhf�c,       ���E	�R[�z�Aԙ*

A2S/average_reward_13��Bt�I$,       ���E	p���z�Aԙ*

A2S/average_reward_1.�B'�:",       ���E	HZ��z�Aԙ*

A2S/average_reward_1���BI�,       ���E	:���z�Aԙ*

A2S/average_reward_1�B�TS�,       ���E	����z�Aԙ*

A2S/average_reward_1�(�B��)",       ���E	<"�z�Aԙ*

A2S/average_reward_13��B�dS,       ���E	7X&�z�Aԙ*

A2S/average_reward_1�L�B��9,       ���E	7�S�z�Aԙ*

A2S/average_reward_1)��B�zl�,       ���E	�u�z�Aԙ*

A2S/average_reward_1ף�B�Y��,       ���E	0f��z�Aԙ*

A2S/average_reward_1�#�BP��g,       ���E	�ڴ�z�Aԙ*

A2S/average_reward_1��Bd\�A,       ���E	c��z�Aԙ*

A2S/average_reward_1���BIJ�,       ���E	�w��z�Aԙ*

A2S/average_reward_1R��B�߰,       ���E	U;�z�Aԙ*

A2S/average_reward_1��BǠ�,       ���E	��A�z�Aԙ*

A2S/average_reward_1���B�i�,       ���E	�a�z�Aԙ*

A2S/average_reward_1��B��A,       ���E	^��z�Aԙ*

A2S/average_reward_1�B��wV,       ���E	̢��z�Aԙ*

A2S/average_reward_1���B&��,       ���E	�Ļ�z�Aԙ*

A2S/average_reward_1R8�B �7�,       ���E	���z�Aԙ*

A2S/average_reward_1
W�B�e,       ���E	���z�Aԙ*

A2S/average_reward_1q=�B^�6,       ���E	S'�z�Aԙ*

A2S/average_reward_1���B%�Rf,       ���E	)�M�z�Aԙ*

A2S/average_reward_1�#�B�7,       ���E	Cpf�z�Aԙ*

A2S/average_reward_1���B(b*,       ���E	/��z�Aԙ*

A2S/average_reward_1�L�B���,       ���E	�ɥ�z�Aԙ*

A2S/average_reward_1R8�B�<s,       ���E	����z�Aԙ*

A2S/average_reward_1��B̙�,       ���E	����z�Aԙ*

A2S/average_reward_1
W�B�6��,       ���E	����z�Aԙ*

A2S/average_reward_1)\�B3�U�,       ���E	��2��z�Aԙ*

A2S/average_reward_1  �B\��C,       ���E	p�W��z�Aԙ*

A2S/average_reward_1  �B���,       ���E	��x��z�Aԙ*

A2S/average_reward_1
��BN�m,       ���E	[N���z�Aԙ*

A2S/average_reward_1��BW%��,       ���E	�ٽ��z�Aԙ*

A2S/average_reward_1.�B� �,       ���E	����z�Aԙ*

A2S/average_reward_1{��BNv�,       ���E	-����z�Aԙ*

A2S/average_reward_1�L�B_"3�,       ���E	�#�z�Aԙ*

A2S/average_reward_1ff�B>���,       ���E	�Q�z�Aԙ*

A2S/average_reward_1�#�B�}��,       ���E	�t�z�Aԙ*

A2S/average_reward_1�#�B�2`�,       ���E	t4��z�Aԙ*

A2S/average_reward_1��Bz��?,       ���E	&-��z�Aԙ*

A2S/average_reward_1.�BKo|�,       ���E	"��z�Aԙ*

A2S/average_reward_1��B�ل�,       ���E	���z�Aԙ*

A2S/average_reward_133�Bڄ�,       ���E	s7%�z�Aԙ*

A2S/average_reward_1ף�B�X��,       ���E	�
O�z�Aԙ*

A2S/average_reward_133�B���,       ���E	�t�z�Aԙ*

A2S/average_reward_1��Bp	�m,       ���E	[��z�Aԙ*

A2S/average_reward_1��B����,       ���E	vf��z�Aԙ*

A2S/average_reward_1���Bj��,       ���E	����z�Aԙ*

A2S/average_reward_1\�B;���,       ���E	�&��z�Aԙ*

A2S/average_reward_1  �B���,       ���E	|��z�Aԙ*

A2S/average_reward_133�B��|,       ���E	�?�z�Aԙ*

A2S/average_reward_1�B�B��ކ,       ���E	�d�z�Aԙ*

A2S/average_reward_1=��B?�b,       ���E	����z�Aԙ*

A2S/average_reward_1�(�B4W{�,       ���E	�!��z�Aԙ*

A2S/average_reward_1���B��n1,       ���E	lD��z�Aԙ*

A2S/average_reward_1q��B��XO,       ���E	}��z�Aԙ*

A2S/average_reward_1  �B�9C�       �v@�	�Y�z�A��	*�

A2S/kl�d=

A2S/average_advantage�Uy�

A2S/policy_network_loss�

A2S/value_network_loss�wA

A2S/q_network_loss��	Aʪx,       ���E	Z�r�z�A��	*

A2S/average_reward_1���B�� ,       ���E	['��z�A��	*

A2S/average_reward_1=��B�)�,       ���E	���z�A��	*

A2S/average_reward_1{��B����,       ���E	�X��z�A��	*

A2S/average_reward_1��B���p,       ���E	g���z�A��	*

A2S/average_reward_1=
�Bכ�,       ���E	��z�A��	*

A2S/average_reward_1
��B��,       ���E	
�7�z�A��	*

A2S/average_reward_1�u�BJL�d,       ���E	�\�z�A��	*

A2S/average_reward_1.�B��6,       ���E	��x�z�A��	*

A2S/average_reward_1Ha�BF�X�,       ���E	∕�z�A��	*

A2S/average_reward_1R8�B�kX,       ���E	���z�A��	*

A2S/average_reward_1R��B��
,       ���E	��z�A��	*

A2S/average_reward_1
��B�A�m,       ���E	�t��z�A��	*

A2S/average_reward_1���B����,       ���E	\��z�A��	*

A2S/average_reward_133�B�&�0,       ���E	��(�z�A��	*

A2S/average_reward_133�BY�,       ���E	b�A�z�A��	*

A2S/average_reward_1���B�W�F,       ���E	G�a�z�A��	*

A2S/average_reward_1=��B*�X#,       ���E	���z�A��	*

A2S/average_reward_1R8�B!�?,       ���E	l��z�A��	*

A2S/average_reward_13��B�W7�,       ���E	���z�A��	*

A2S/average_reward_1�B��|�,       ���E	���z�A��	*

A2S/average_reward_1{��BUD��,       ���E	����z�A��	*

A2S/average_reward_1���B����,       ���E	���z�A��	*

A2S/average_reward_1q=�B��{�,       ���E	��A��z�A��	*

A2S/average_reward_1R8�B`C۾,       ���E	r�a��z�A��	*

A2S/average_reward_1��B���,       ���E	��|��z�A��	*

A2S/average_reward_1ff�Bs%C,       ���E	����z�A��	*

A2S/average_reward_1���B��OD,       ���E	x5���z�A��	*

A2S/average_reward_1ף�B�w�,       ���E	'���z�A��	*

A2S/average_reward_1H��B%<G�,       ���E	���z�A��	*

A2S/average_reward_1�p�B�Y�,       ���E	�l!��z�A��	*

A2S/average_reward_1���B���b,       ���E	�NE��z�A��	*

A2S/average_reward_1=��B¾��,       ���E	u�j��z�A��	*

A2S/average_reward_1�#�B���:,       ���E	z���z�A��	*

A2S/average_reward_1.�B��G{,       ���E	�G���z�A��	*

A2S/average_reward_1���B@cx,       ���E	\����z�A��	*

A2S/average_reward_1.�B��u�,       ���E	T����z�A��	*

A2S/average_reward_1
W�BWr9T,       ���E	c3��z�A��	*

A2S/average_reward_1)\�B����,       ���E	���z�A��	*

A2S/average_reward_1H��B�,       ���E	tm@��z�A��	*

A2S/average_reward_1  �BȘN ,       ���E	P�[��z�A��	*

A2S/average_reward_1�u�B��i�,       ���E	�*x��z�A��	*

A2S/average_reward_1��B�-̣,       ���E	&t���z�A��	*

A2S/average_reward_1���B-��,       ���E	FZ���z�A��	*

A2S/average_reward_1
��B���,       ���E	�2���z�A��	*

A2S/average_reward_1���B	�/:,       ���E	ѵ���z�A��	*

A2S/average_reward_13��B00�,       ���E	O� ��z�A��	*

A2S/average_reward_1���Bi(�,       ���E	�!��z�A��	*

A2S/average_reward_1���B��7
,       ���E	ځB��z�A��	*

A2S/average_reward_1=��B���,       ���E	\i\��z�A��	*

A2S/average_reward_1�Q�BX�}F,       ���E	x��z�A��	*

A2S/average_reward_1ף�B��#�,       ���E	8���z�A��	*

A2S/average_reward_1q��B�9G,       ���E	첷��z�A��	*

A2S/average_reward_1ff�B�=5�,       ���E	����z�A��	*

A2S/average_reward_1ף�Bt��,       ���E	�� ��z�A��	*

A2S/average_reward_1\�B �5�,       ���E	���z�A��	*

A2S/average_reward_1���B�fd�,       ���E	dC;��z�A��	*

A2S/average_reward_1
��B�R,       ���E	g_T��z�A��	*

A2S/average_reward_1�B�Bf���,       ���E	s�z��z�A��	*

A2S/average_reward_1��B'*��,       ���E	m'���z�A��	*

A2S/average_reward_1=��B�,,       ���E	cҲ��z�A��	*

A2S/average_reward_1ff�B�+L	,       ���E	����z�A��	*

A2S/average_reward_1Ha�Bxp̌,       ���E	�����z�A��	*

A2S/average_reward_1�B�B �\�,       ���E	g��z�A��	*

A2S/average_reward_133�B�L�,       ���E	�:��z�A��	*

A2S/average_reward_1�k�B ���,       ���E	l�_��z�A��	*

A2S/average_reward_1���Bs�,       ���E	����z�A��	*

A2S/average_reward_1�Q�B�K�C,       ���E	����z�A��	*

A2S/average_reward_1�#�B���6,       ���E	I����z�A��	*

A2S/average_reward_1q��B�-B,       ���E	�i���z�A��	*

A2S/average_reward_1Ha�B͐�:,       ���E	4���z�A��	*

A2S/average_reward_1�B�B�B�A,       ���E	h�-��z�A��	*

A2S/average_reward_1{�B��C�,       ���E	^XN��z�A��	*

A2S/average_reward_1f��BSL��,       ���E	�g��z�A��	*

A2S/average_reward_1=
�B�"3,       ���E	U����z�A��	*

A2S/average_reward_1���BB�چ,       ���E	R%���z�A��	*

A2S/average_reward_1���B�C�9,       ���E	�w���z�A��	*

A2S/average_reward_1���B��,       ���E	!���z�A��	*

A2S/average_reward_13��B<��,       ���E	��
��z�A��	*

A2S/average_reward_1�L�Bo��,       ���E	3�2��z�A��	*

A2S/average_reward_1���B/���,       ���E	R�[��z�A��	*

A2S/average_reward_1�G�B{ɂ,       ���E	i�v��z�A��	*

A2S/average_reward_1Ha�B����,       ���E	%]���z�A��	*

A2S/average_reward_1���B�9��,       ���E	�|���z�A��	*

A2S/average_reward_1�z�B�u,       ���E	���z�A��	*

A2S/average_reward_1=
�B|ϙ,       ���E	h���z�A��	*

A2S/average_reward_1
W�B���,       ���E	����z�A��	*

A2S/average_reward_1)\�B�;��,       ���E	��1��z�A��	*

A2S/average_reward_1ff�BD�
�,       ���E	�O��z�A��	*

A2S/average_reward_1���B��{ ,       ���E	��s��z�A��	*

A2S/average_reward_1H��Bqw,       ���E	S���z�A��	*

A2S/average_reward_1�k�B$@I�,       ���E	�o���z�A��	*

A2S/average_reward_1�k�B�%,       ���E	^����z�A��	*

A2S/average_reward_1���B��0,       ���E	�%���z�A��	*

A2S/average_reward_1�L�B���i,       ���E	%���z�A��	*

A2S/average_reward_1���B�$�Y,       ���E	�f��z�A��	*

A2S/average_reward_1)\�B��,       ���E	$:��z�A��	*

A2S/average_reward_1���Bt��,       ���E	��\��z�A��	*

A2S/average_reward_1���Bߙ��,       ���E	��~��z�A��	*

A2S/average_reward_1���B!<
,       ���E	M���z�A��	*

A2S/average_reward_1���B�4,       ���E	�����z�A��	*

A2S/average_reward_13��B:U,       ���E	�����z�A��	*

A2S/average_reward_1���B��?�,       ���E	�����z�A��	*

A2S/average_reward_1�z�B-�l,       ���E	W��z�A��	*

A2S/average_reward_1=
�BU'<�,       ���E	��9��z�A��	*

A2S/average_reward_1{�B���,       ���E	��S��z�A��	*

A2S/average_reward_1ף�B��i|,       ���E	�5y��z�A��	*

A2S/average_reward_1=��BVa,       ���E	�D���z�A��	*

A2S/average_reward_13��B3c�,       ���E	e*���z�A��	*

A2S/average_reward_1  �B�9�3,       ���E	Q����z�A��	*

A2S/average_reward_1�z�B�>k,       ���E	ҿ���z�A��	*

A2S/average_reward_1�B�Bt8P[,       ���E	�	��z�A��	*

A2S/average_reward_1�(�B֡r�,       ���E	u ��z�A��	*

A2S/average_reward_133�B/�P,       ���E	��;��z�A��	*

A2S/average_reward_1{�B1�,       ���E	�Y��z�A��	*

A2S/average_reward_1�u�B'�n,       ���E	#�z��z�A��	*

A2S/average_reward_1�(�B��9,       ���E	�����z�A��	*

A2S/average_reward_1�L�BY�Z,       ���E	�����z�A��	*

A2S/average_reward_1�L�B��`�,       ���E	����z�A��	*

A2S/average_reward_1�z�B�=F,       ���E	� ���z�A��	*

A2S/average_reward_1���B|qx�,       ���E	J �z�A��	*

A2S/average_reward_1f��B��h�,       ���E	A�# �z�A��	*

A2S/average_reward_1\�B�X��,       ���E	җB �z�A��	*

A2S/average_reward_1�(�BJPε,       ���E	��c �z�A��	*

A2S/average_reward_1 ��B7lK,       ���E	�� �z�A��	*

A2S/average_reward_1R��B���,       ���E	�A� �z�A��	*

A2S/average_reward_1
��B|I9|,       ���E	�� �z�A��	*

A2S/average_reward_1�k�B�d`�,       ���E	�� �z�A��	*

A2S/average_reward_1���BVn��,       ���E	*9�z�A��	*

A2S/average_reward_1{�B�!��,       ���E	��(�z�A��	*

A2S/average_reward_1\�B�\�,       ���E	��8�z�A��	*

A2S/average_reward_1�G�B��	u,       ���E		�U�z�A��	*

A2S/average_reward_1���B��!�,       ���E	��s�z�A��	*

A2S/average_reward_1�Q�B+�x�,       ���E	����z�A��	*

A2S/average_reward_1��B+Q�,       ���E	Ws��z�A��	*

A2S/average_reward_1R8�B��,       ���E	�U��z�A��	*

A2S/average_reward_1
W�B'�S,       ���E	���z�A��	*

A2S/average_reward_1ף�B�t ,       ���E	�l%�z�A��	*

A2S/average_reward_1�z�B����,       ���E	
F�z�A��	*

A2S/average_reward_1{��Bog��,       ���E	T�m�z�A��	*

A2S/average_reward_1\��B�I��,       ���E	`���z�A��	*

A2S/average_reward_133�B�u֊,       ���E	�'��z�A��	*

A2S/average_reward_1���B�4$p,       ���E	�I��z�A��	*

A2S/average_reward_1�#�B���,       ���E	����z�A��	*

A2S/average_reward_1��B!�q,       ���E	����z�A��	*

A2S/average_reward_1
��B�z�,       ���E	���z�A��	*

A2S/average_reward_1�#�B���,       ���E	_O<�z�A��	*

A2S/average_reward_1
��B�G�(,       ���E	!�_�z�A��	*

A2S/average_reward_1���B+"�,       ���E	��~�z�A��	*

A2S/average_reward_1{��Bld,       ���E	�m��z�A��	*

A2S/average_reward_1q=�Ba�",       ���E	(��z�A��	*

A2S/average_reward_1���BX*w�,       ���E	|��z�A��	*

A2S/average_reward_1��BM	��,       ���E	v�z�A��	*

A2S/average_reward_1�(�Br�?,       ���E	�L(�z�A��	*

A2S/average_reward_1q=�B�'�,       ���E	4I�z�A��	*

A2S/average_reward_1�(�B���=,       ���E	k.b�z�A��	*

A2S/average_reward_1�G�B-r�,       ���E	���z�A��	*

A2S/average_reward_1H��BX�,       ���E	S͞�z�A��	*

A2S/average_reward_133�B+�:Y,       ���E	�%��z�A��	*

A2S/average_reward_1f��B�n�L,       ���E	I���z�A��	*

A2S/average_reward_1{��B9C�,       ���E	����z�A��	*

A2S/average_reward_1{��B3�	k,       ���E	1��z�A��	*

A2S/average_reward_1 ��Bݲ�,       ���E	�e9�z�A��	*

A2S/average_reward_1�z�B5���,       ���E	�O�z�A��	*

A2S/average_reward_1���B��X�,       ���E	��i�z�A��	*

A2S/average_reward_1)��B싡�,       ���E	�^��z�A��	*

A2S/average_reward_1�BpN�$,       ���E	h{��z�A��	*

A2S/average_reward_1{��B�2kq,       ���E	��z�A��	*

A2S/average_reward_1���B�4t�,       ���E	�T��z�A��	*

A2S/average_reward_1�B���,       ���E	��z�A��	*

A2S/average_reward_1
W�B]�S,       ���E	FU �z�A��	*

A2S/average_reward_1�z�B��b�,       ���E	&8H�z�A��	*

A2S/average_reward_1�p�B��,       ���E	�g�z�A��	*

A2S/average_reward_1 ��BśV,       ���E	2��z�A��	*

A2S/average_reward_1  �B01g�,       ���E	����z�A��	*

A2S/average_reward_1�p�B��~,       ���E	����z�A��	*

A2S/average_reward_1�B<ݶ,       ���E	���z�A��	*

A2S/average_reward_1�k�B�j��,       ���E	7j�z�A��	*

A2S/average_reward_1=��B�B�,       ���E	��z�A��	*

A2S/average_reward_1���B�+"�,       ���E	��;�z�A��	*

A2S/average_reward_1{�Bvws��       �v@�	����z�A��
*�

A2S/klny==

A2S/average_advantage�-?

A2S/policy_network_loss�D�>

A2S/value_network_loss#�A

A2S/q_network_loss.��A�\S�,       ���E	���z�A��
*

A2S/average_reward_1���B�B��,       ���E	����z�A��
*

A2S/average_reward_1��B�eR�,       ���E	�6 �z�A��
*

A2S/average_reward_1��B�[Y�,       ���E	�F�z�A��
*

A2S/average_reward_1{��B{��H,       ���E	�6�z�A��
*

A2S/average_reward_1Ha�BBު,       ���E	;�R�z�A��
*

A2S/average_reward_1q��B��Q�,       ���E	�0x�z�A��
*

A2S/average_reward_1ff�B�L��,       ���E	�Ô�z�A��
*

A2S/average_reward_1�B�B��%,       ���E	���z�A��
*

A2S/average_reward_1���B�>�,       ���E	����z�A��
*

A2S/average_reward_1=
�B���,       ���E	����z�A��
*

A2S/average_reward_1{��B.�6�,       ���E	�8	�z�A��
*

A2S/average_reward_1)\�B�R ,       ���E	H�=	�z�A��
*

A2S/average_reward_1)\�B�k�,       ���E	��V	�z�A��
*

A2S/average_reward_1��B1�C�,       ���E	w*v	�z�A��
*

A2S/average_reward_1=��BM�jc,       ���E	�|�	�z�A��
*

A2S/average_reward_1�u�BS��,       ���E	��	�z�A��
*

A2S/average_reward_1�z�BL�,       ���E	�L�	�z�A��
*

A2S/average_reward_1�#�B� ,       ���E	{g�	�z�A��
*

A2S/average_reward_1
��B*ƶ�,       ���E	�e
�z�A��
*

A2S/average_reward_1���B�Ql�,       ���E	%�'
�z�A��
*

A2S/average_reward_1���B��>�,       ���E	�F
�z�A��
*

A2S/average_reward_1���B0s�i,       ���E	�'b
�z�A��
*

A2S/average_reward_1q=�BY K,       ���E	�_}
�z�A��
*

A2S/average_reward_1{�B(-D,       ���E	���
�z�A��
*

A2S/average_reward_1)\�Bk�^,       ���E	�(�
�z�A��
*

A2S/average_reward_1���B��}W,       ���E	��
�z�A��
*

A2S/average_reward_1���B��9,       ���E	���
�z�A��
*

A2S/average_reward_1��Bf^�~,       ���E	�R�z�A��
*

A2S/average_reward_1q��B�?�`,       ���E	H�'�z�A��
*

A2S/average_reward_1H��B-&��,       ���E	�C�z�A��
*

A2S/average_reward_1�L�B:"��,       ���E	n�g�z�A��
*

A2S/average_reward_1��Bs�.,       ���E	����z�A��
*

A2S/average_reward_1��BGԒ�,       ���E	R	��z�A��
*

A2S/average_reward_1���B4��,       ���E	���z�A��
*

A2S/average_reward_1{��B�9�A,       ���E	yu��z�A��
*

A2S/average_reward_1f��B��,       ���E	zc��z�A��
*

A2S/average_reward_1
��Bxd�,       ���E	�a�z�A��
*

A2S/average_reward_1��B�A�m,       ���E	��9�z�A��
*

A2S/average_reward_1��B���,       ���E	�E\�z�A��
*

A2S/average_reward_1���B�+O�,       ���E	�}�z�A��
*

A2S/average_reward_1���Bz#X,       ���E	C��z�A��
*

A2S/average_reward_1f��B�K8,       ���E	�g��z�A��
*

A2S/average_reward_13��BI�{,       ���E	����z�A��
*

A2S/average_reward_1H��By�`,       ���E	^���z�A��
*

A2S/average_reward_1q=�B�|�a,       ���E	hZ�z�A��
*

A2S/average_reward_1���B�t�,       ���E	���z�A��
*

A2S/average_reward_1)��Bk���,       ���E	[&=�z�A��
*

A2S/average_reward_1  �B��ɔ,       ���E	 1[�z�A��
*

A2S/average_reward_1���Bಉ5,       ���E	Txy�z�A��
*

A2S/average_reward_1\��B�͝},       ���E	�B��z�A��
*

A2S/average_reward_1 ��B� 2�,       ���E	�L��z�A��
*

A2S/average_reward_1�p�B��+Y,       ���E	Ѵ��z�A��
*

A2S/average_reward_1ף�BS�`�,       ���E	���z�A��
*

A2S/average_reward_1�L�B
�*�,       ���E	$,�z�A��
*

A2S/average_reward_1)\�B�C�h,       ���E	�,�z�A��
*

A2S/average_reward_1H��B�H,       ���E	q�L�z�A��
*

A2S/average_reward_1���B]�n,       ���E	�fc�z�A��
*

A2S/average_reward_1f��B{�q
,       ���E	��w�z�A��
*

A2S/average_reward_1Ha�Bh�;,       ���E	�4��z�A��
*

A2S/average_reward_1���B\vI�,       ���E	��z�A��
*

A2S/average_reward_1�Q�B�W,,       ���E	C4��z�A��
*

A2S/average_reward_1���BN��U,       ���E	G���z�A��
*

A2S/average_reward_1��B�|�,       ���E	�d��z�A��
*

A2S/average_reward_1��B��gB,       ���E	���z�A��
*

A2S/average_reward_1ff�B?Q]9,       ���E	�)?�z�A��
*

A2S/average_reward_1q=�B.��,       ���E	TZ�z�A��
*

A2S/average_reward_1���B�e�,       ���E	�u�z�A��
*

A2S/average_reward_1��B�<2,       ���E	�D��z�A��
*

A2S/average_reward_133�B�'R�,       ���E	3���z�A��
*

A2S/average_reward_1���BT��v,       ���E	�'��z�A��
*

A2S/average_reward_1�p�B>��p,       ���E	�W��z�A��
*

A2S/average_reward_1�L�Bh�̧,       ���E	��z�A��
*

A2S/average_reward_1ף�B�d y,       ���E	�(�z�A��
*

A2S/average_reward_133�B^x�,       ���E	b4A�z�A��
*

A2S/average_reward_1 ��B�e�4,       ���E	�_�z�A��
*

A2S/average_reward_1R��B���,       ���E	T���z�A��
*

A2S/average_reward_1���B9�,       ���E	����z�A��
*

A2S/average_reward_1�B�^,       ���E	*���z�A��
*

A2S/average_reward_1q=�B
f�,       ���E	$���z�A��
*

A2S/average_reward_1���B�X��,       ���E	#b��z�A��
*

A2S/average_reward_1���BU�.c,       ���E	%��z�A��
*

A2S/average_reward_1f��BAB�,       ���E	 �6�z�A��
*

A2S/average_reward_1R��B~8�,       ���E	��R�z�A��
*

A2S/average_reward_1���Bh9Y,       ���E	r�g�z�A��
*

A2S/average_reward_1Ha�B�i,       ���E	���z�A��
*

A2S/average_reward_1���B�k,       ���E	 $��z�A��
*

A2S/average_reward_1{�B�؆,       ���E	���z�A��
*

A2S/average_reward_1)\�B��!,       ���E	�
��z�A��
*

A2S/average_reward_1ף�B�a�R,       ���E	!���z�A��
*

A2S/average_reward_1���BiᎿ,       ���E	k3�z�A��
*

A2S/average_reward_1�B�Bu��,       ���E	�-/�z�A��
*

A2S/average_reward_1Ha�Bmd��,       ���E	�IQ�z�A��
*

A2S/average_reward_1��B���,       ���E	:u�z�A��
*

A2S/average_reward_1���B=;Q,       ���E	���z�A��
*

A2S/average_reward_1\�B�%��,       ���E	���z�A��
*

A2S/average_reward_1�(�B(f`m,       ���E	h��z�A��
*

A2S/average_reward_1=��B�H��,       ���E	S_��z�A��
*

A2S/average_reward_1=
�B����,       ���E	���z�A��
*

A2S/average_reward_1q=�B�W��,       ���E	���z�A��
*

A2S/average_reward_1  �B���i,       ���E	��<�z�A��
*

A2S/average_reward_1�#�B|,       ���E	JVT�z�A��
*

A2S/average_reward_1=��B�D��,       ���E	C�l�z�A��
*

A2S/average_reward_1�u�By���,       ���E	k2��z�A��
*

A2S/average_reward_1���B���,       ���E	�g��z�A��
*

A2S/average_reward_1\��B ���,       ���E	����z�A��
*

A2S/average_reward_1���B�fc�,       ���E	���z�A��
*

A2S/average_reward_1q��B0���,       ���E	�(��z�A��
*

A2S/average_reward_1\�B��m,       ���E	7�z�A��
*

A2S/average_reward_1�#�BDE�,       ���E	�{&�z�A��
*

A2S/average_reward_1.�BZE$,       ���E	��A�z�A��
*

A2S/average_reward_1���B�w��,       ���E	��d�z�A��
*

A2S/average_reward_1�B`�m,       ���E	)���z�A��
*

A2S/average_reward_1���B�|��,       ���E	����z�A��
*

A2S/average_reward_1{��B��J,       ���E	L��z�A��
*

A2S/average_reward_1��B����,       ���E	����z�A��
*

A2S/average_reward_1�p�B;���,       ���E	����z�A��
*

A2S/average_reward_1R8�B���,       ���E	X\�z�A��
*

A2S/average_reward_1�k�B*�	�,       ���E	��'�z�A��
*

A2S/average_reward_1�k�B]��,       ���E	"�?�z�A��
*

A2S/average_reward_1�G�BU�a,       ���E	W�]�z�A��
*

A2S/average_reward_1
W�B���",       ���E	e΀�z�A��
*

A2S/average_reward_1Ha�B�e�,       ���E	Y���z�A��
*

A2S/average_reward_1=��B˾2T,       ���E	xI��z�A��
*

A2S/average_reward_1���B�{,       ���E	�,��z�A��
*

A2S/average_reward_1{��BC���,       ���E	����z�A��
*

A2S/average_reward_1�G�Bn�c,       ���E	;��z�A��
*

A2S/average_reward_1�#�B�I�:,       ���E	�3�z�A��
*

A2S/average_reward_1�(�Bc�l,       ���E	g�S�z�A��
*

A2S/average_reward_1R8�B��߀,       ���E	Ino�z�A��
*

A2S/average_reward_1�B�B���,       ���E	����z�A��
*

A2S/average_reward_1�u�Be��7,       ���E	a��z�A��
*

A2S/average_reward_1R8�B8?8',       ���E	���z�A��
*

A2S/average_reward_1R8�B��m,       ���E	���z�A��
*

A2S/average_reward_1f��B &,       ���E	�r��z�A��
*

A2S/average_reward_1f��Bb��,       ���E	8�	�z�A��
*

A2S/average_reward_1���B�,       ���E	ũ!�z�A��
*

A2S/average_reward_1���B� #�,       ���E	�;>�z�A��
*

A2S/average_reward_1f��B?���,       ���E	�]X�z�A��
*

A2S/average_reward_1���B�)^�,       ���E	��s�z�A��
*

A2S/average_reward_1=��B[�(`,       ���E	8���z�A��
*

A2S/average_reward_1=��Bt���,       ���E	����z�A��
*

A2S/average_reward_1�G�B�̐�,       ���E	B��z�A��
*

A2S/average_reward_1�u�B̋*�,       ���E	n���z�A��
*

A2S/average_reward_1  �BJ�?,       ���E	���z�A��
*

A2S/average_reward_1��B�,P,       ���E	C<3�z�A��
*

A2S/average_reward_1f��B�]y,       ���E	�O�z�A��
*

A2S/average_reward_1�u�Bb�az,       ���E	]oo�z�A��
*

A2S/average_reward_1���B����,       ���E	Ғ��z�A��
*

A2S/average_reward_1���BpNS�,       ���E	��z�A��
*

A2S/average_reward_1H��B>��5,       ���E	6#��z�A��
*

A2S/average_reward_1=
�B"�sL,       ���E	���z�A��
*

A2S/average_reward_1�B}�GR,       ���E	@��z�A��
*

A2S/average_reward_1)��B��j,       ���E	��+�z�A��
*

A2S/average_reward_1���B�;�/,       ���E	��H�z�A��
*

A2S/average_reward_1 ��Bs�I,       ���E	w(^�z�A��
*

A2S/average_reward_1�p�BM���,       ���E	L�u�z�A��
*

A2S/average_reward_1
W�B�j~,       ���E	G2��z�A��
*

A2S/average_reward_1H��B�p-,       ���E	M���z�A��
*

A2S/average_reward_1Ha�B��,       ���E	4���z�A��
*

A2S/average_reward_1\�B�>�u,       ���E	����z�A��
*

A2S/average_reward_1ff�B�y[-,       ���E	���z�A��
*

A2S/average_reward_1�k�B��9,       ���E	CU'�z�A��
*

A2S/average_reward_1
��B��
,       ���E	F/F�z�A��
*

A2S/average_reward_1���B�7�+,       ���E	��^�z�A��
*

A2S/average_reward_1���BGU�,       ���E	�,w�z�A��
*

A2S/average_reward_1�u�B�� ,       ���E	�C��z�A��
*

A2S/average_reward_1q��B�e�h,       ���E	7��z�A��
*

A2S/average_reward_1R��B����,       ���E	$���z�A��
*

A2S/average_reward_1�p�B}��:,       ���E	����z�A��
*

A2S/average_reward_1�(�B݃�,       ���E	P���z�A��
*

A2S/average_reward_1.�B���,       ���E	��z�A��
*

A2S/average_reward_1
��B�w
L,       ���E	,�z�A��
*

A2S/average_reward_1
��B���,       ���E	��G�z�A��
*

A2S/average_reward_1
��B���,       ���E	^e�z�A��
*

A2S/average_reward_1�L�BL}o,       ���E	�N��z�A��
*

A2S/average_reward_1{�B���,       ���E	П��z�A��
*

A2S/average_reward_1  �B\pO,       ���E	E���z�A��
*

A2S/average_reward_1���B��ѝ,       ���E	EX��z�A��
*

A2S/average_reward_133�BX��,       ���E	m��z�A��
*

A2S/average_reward_13��B�͕,       ���E	�&�z�A��
*

A2S/average_reward_1R��B�t=�,       ���E	m�F�z�A��
*

A2S/average_reward_1q��B���,       ���E	c�c�z�A��
*

A2S/average_reward_1��B��S�,       ���E	+�{�z�A��
*

A2S/average_reward_1���B+V$:,       ���E	k��z�A��
*

A2S/average_reward_1=
�B�p؎,       ���E	K��z�A��
*

A2S/average_reward_1���B��~,       ���E		���z�A��
*

A2S/average_reward_1.�BeD*',       ���E	�)�z�A��
*

A2S/average_reward_1)\�B���a,       ���E	mT�z�A��
*

A2S/average_reward_1���B>q��,       ���E	q�1�z�A��
*

A2S/average_reward_1H��B���,       ���E	�G�z�A��
*

A2S/average_reward_1��B-=�,       ���E	��b�z�A��
*

A2S/average_reward_1\��B��
,       ���E	U��z�A��
*

A2S/average_reward_1��Bn�u�,       ���E	����z�A��
*

A2S/average_reward_1ff�B��t�,       ���E	����z�A��
*

A2S/average_reward_1���B8�o�,       ���E	��z�A��
*

A2S/average_reward_1q��B�u/�,       ���E	����z�A��
*

A2S/average_reward_1�L�B8�b�,       ���E	��z�A��
*

A2S/average_reward_1���BiG�
,       ���E	h�/�z�A��
*

A2S/average_reward_1���B��)},       ���E	�P�z�A��
*

A2S/average_reward_1H��B����       �v@�	�ݺ�z�A��*�

A2S/kl(�2=

A2S/average_advantage#�?=

A2S/policy_network_loss�=

A2S/value_network_lossg*@

A2S/q_network_loss�� @ߓ_�,       ���E	����z�A��*

A2S/average_reward_1���Bt�;S,       ���E	���z�A��*

A2S/average_reward_1��B��3�,       ���E	+�
�z�A��*

A2S/average_reward_1{�B���X,       ���E	�,�z�A��*

A2S/average_reward_1��B���=,       ���E	+�F�z�A��*

A2S/average_reward_1���B�E]�,       ���E	�"e�z�A��*

A2S/average_reward_1�(�BJBM,       ���E	�z�A��*

A2S/average_reward_1H��BP��r,       ���E	�Ԥ�z�A��*

A2S/average_reward_1���B$v�,       ���E	���z�A��*

A2S/average_reward_1q��B�Ԣ�,       ���E	f���z�A��*

A2S/average_reward_1���B��j�,       ���E	JC��z�A��*

A2S/average_reward_1���Buc�;,       ���E	25 �z�A��*

A2S/average_reward_1��BGv�,       ���E	5�9 �z�A��*

A2S/average_reward_1H��B��B�,       ���E	�b �z�A��*

A2S/average_reward_1���B���,       ���E	>{ �z�A��*

A2S/average_reward_1�k�B�i�,       ���E	�}� �z�A��*

A2S/average_reward_1\�Bo��,       ���E	� �z�A��*

A2S/average_reward_1���B�0�;,       ���E	=�� �z�A��*

A2S/average_reward_1���B��q|,       ���E	�!�z�A��*

A2S/average_reward_1{��B�[y,       ���E	S�!�z�A��*

A2S/average_reward_1\��BS�Lx,       ���E	ȹ8!�z�A��*

A2S/average_reward_1���B8���,       ���E	�X!�z�A��*

A2S/average_reward_1���Bwf,       ���E	���!�z�A��*

A2S/average_reward_1.�B���-,       ���E	��!�z�A��*

A2S/average_reward_1�z�BT��,       ���E	�K�!�z�A��*

A2S/average_reward_1q��Bp�_.,       ���E	��!�z�A��*

A2S/average_reward_1H��BeF�*,       ���E	Pj"�z�A��*

A2S/average_reward_1�Bz�ݒ,       ���E	��/"�z�A��*

A2S/average_reward_1)\�B�VQ,       ���E	}bO"�z�A��*

A2S/average_reward_1���B��w�,       ���E	�4o"�z�A��*

A2S/average_reward_1���B���C,       ���E	3t�"�z�A��*

A2S/average_reward_13��B��M�,       ���E	jԱ"�z�A��*

A2S/average_reward_1���B{?�,       ���E	�Q�"�z�A��*

A2S/average_reward_1=��B��E,       ���E	`��"�z�A��*

A2S/average_reward_1=��B/O]n,       ���E	���"�z�A��*

A2S/average_reward_1�k�B<���,       ���E	�R"#�z�A��*

A2S/average_reward_1���B.�,       ���E	��@#�z�A��*

A2S/average_reward_1H��B۹E,       ���E	%�`#�z�A��*

A2S/average_reward_1H��Bz�J,       ���E	�x�#�z�A��*

A2S/average_reward_1{�B:�r~,       ���E	�#�z�A��*

A2S/average_reward_1)��BNm��,       ���E	�b�#�z�A��*

A2S/average_reward_1{��B��u,       ���E	�h�#�z�A��*

A2S/average_reward_1���B�1��,       ���E	Hj�#�z�A��*

A2S/average_reward_1��B��_`,       ���E	ж$�z�A��*

A2S/average_reward_1\�B�}"B,       ���E	��1$�z�A��*

A2S/average_reward_1.�B�>�*,       ���E	��N$�z�A��*

A2S/average_reward_1���By��n,       ���E	�q$�z�A��*

A2S/average_reward_1H��BQ�j�,       ���E	��$�z�A��*

A2S/average_reward_1
��B��C,       ���E	L]�$�z�A��*

A2S/average_reward_133�B�-,       ���E	]��$�z�A��*

A2S/average_reward_1��B�y�,       ���E	�b�$�z�A��*

A2S/average_reward_1�G�B\H��,       ���E	 �#%�z�A��*

A2S/average_reward_1H��B�L��,       ���E	>D%�z�A��*

A2S/average_reward_1
��B:Ɠ�,       ���E	�"a%�z�A��*

A2S/average_reward_1�u�B_��m,       ���E	���%�z�A��*

A2S/average_reward_1���Bڷ�e,       ���E	��%�z�A��*

A2S/average_reward_1��BΟ��,       ���E	5@�%�z�A��*

A2S/average_reward_1q��B�4�,       ���E	�_�%�z�A��*

A2S/average_reward_1�L�B^��,       ���E	�m&�z�A��*

A2S/average_reward_1�B�B�V�,       ���E	\�$&�z�A��*

A2S/average_reward_1��B鏔�,       ���E	��@&�z�A��*

A2S/average_reward_1���Bk^�S,       ���E	Q:`&�z�A��*

A2S/average_reward_1�u�B ?��,       ���E	�nw&�z�A��*

A2S/average_reward_1���BL~�j,       ���E	h�&�z�A��*

A2S/average_reward_1=
�B?h�6,       ���E	�&�z�A��*

A2S/average_reward_1H��B7��,       ���E	�&�z�A��*

A2S/average_reward_1�Q�Bz#ʫ,       ���E	~��&�z�A��*

A2S/average_reward_1���B���w,       ���E	�'�z�A��*

A2S/average_reward_1.�B�),,       ���E	T@'�z�A��*

A2S/average_reward_1���B塝;,       ���E	R<^'�z�A��*

A2S/average_reward_1��B����,       ���E	��z'�z�A��*

A2S/average_reward_1�(�B�(�,       ���E	�'�z�A��*

A2S/average_reward_1q=�B�D,       ���E	��'�z�A��*

A2S/average_reward_1�B����,       ���E	X��'�z�A��*

A2S/average_reward_1�k�B_i��,       ���E	NX�'�z�A��*

A2S/average_reward_1���B�,       ���E	��'(�z�A��*

A2S/average_reward_1 ��B��.�,       ���E	�ML(�z�A��*

A2S/average_reward_1R��B9��,       ���E	�c(�z�A��*

A2S/average_reward_1ff�B&�G,       ���E	�}(�z�A��*

A2S/average_reward_1���B�H"-,       ���E	���(�z�A��*

A2S/average_reward_1���B��M0,       ���E	��(�z�A��*

A2S/average_reward_1ff�B;��,       ���E	���(�z�A��*

A2S/average_reward_1��B���,       ���E	j�)�z�A��*

A2S/average_reward_1���B�0�,       ���E	IC*)�z�A��*

A2S/average_reward_1\�B�O��,       ���E	Q�I)�z�A��*

A2S/average_reward_133�B��!,       ���E	��f)�z�A��*

A2S/average_reward_1R8�B˩	,       ���E	�D�)�z�A��*

A2S/average_reward_1���B� ,       ���E	c��)�z�A��*

A2S/average_reward_133�BP�-,       ���E	��)�z�A��*

A2S/average_reward_1���B�L�1,       ���E	���)�z�A��*

A2S/average_reward_1f��B�FO,       ���E	��*�z�A��*

A2S/average_reward_13��B��,       ���E	��,*�z�A��*

A2S/average_reward_1{��B�M",       ���E	s�G*�z�A��*

A2S/average_reward_1�z�B�qC�,       ���E	�d*�z�A��*

A2S/average_reward_1q��B�a.�,       ���E	V�z*�z�A��*

A2S/average_reward_1�k�B4K��,       ���E	)X�*�z�A��*

A2S/average_reward_1���B��Y�,       ���E	�C�*�z�A��*

A2S/average_reward_1��B��q#,       ���E	]S�*�z�A��*

A2S/average_reward_1  �Bj2|�,       ���E	�� +�z�A��*

A2S/average_reward_1=��B�4��,       ���E	�m+�z�A��*

A2S/average_reward_1��B� 6u,       ���E	~�;+�z�A��*

A2S/average_reward_1 ��B�9�4,       ���E	۲]+�z�A��*

A2S/average_reward_1���B\��q,       ���E	P'{+�z�A��*

A2S/average_reward_1���BlQ�,       ���E	j#�+�z�A��*

A2S/average_reward_1.�B�cM�,       ���E	��+�z�A��*

A2S/average_reward_1
��B���,       ���E	f��+�z�A��*

A2S/average_reward_1ף�Bz�!V,       ���E	��+�z�A��*

A2S/average_reward_1{��B	Cr,       ���E	,�z�A��*

A2S/average_reward_1�B��d3,       ���E	>i:,�z�A��*

A2S/average_reward_1�B�B��)