       �K"	  @?�z�Abrain.Event:2�p	ޕ;     ��_�	cJ?�z�A"��
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
PA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:*
T0
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
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
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
<A2S/backup_policy_network/backup_policy_network/fc0/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/bGA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zeros*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
:A2S/backup_policy_network/backup_policy_network/fc0/b/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
_output_shapes
:
�
 A2S/backup_policy_network/MatMulMatMulA2S/observations:A2S/backup_policy_network/backup_policy_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
A2S/backup_policy_network/addAdd A2S/backup_policy_network/MatMul:A2S/backup_policy_network/backup_policy_network/fc0/b/read*
T0*'
_output_shapes
:���������
�
:A2S/backup_policy_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
valueB*    
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
	container *
shape:
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
BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
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
4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
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
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
�
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
5A2S/backup_policy_network/backup_policy_network/out/w
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
	container 
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
VariableV2*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
	container *
shape:*
dtype0
�
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"      
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
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
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
8A2S/best_policy_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0
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
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
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
.A2S/best_policy_network/LayerNorm/moments/meanMeanA2S/best_policy_network/add@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
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
2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
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
/A2S/best_policy_network/LayerNorm/batchnorm/subSub+A2S/best_policy_network/LayerNorm/beta/read1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
�
1A2S/best_policy_network/LayerNorm/batchnorm/add_1Add1A2S/best_policy_network/LayerNorm/batchnorm/mul_1/A2S/best_policy_network/LayerNorm/batchnorm/sub*'
_output_shapes
:���������*
T0
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
seed2u*
dtype0*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
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
_output_shapes

:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:*
dtype0
�
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
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
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/add_16A2S/best_policy_network/best_policy_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shape*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:*

seed
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
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
	container 
�
:A2S/backup_value_network/backup_value_network/fc0/w/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/wNA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
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
VariableV2*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(*
_output_shapes
:
�
8A2S/backup_value_network/backup_value_network/fc0/b/readIdentity3A2S/backup_value_network/backup_value_network/fc0/b*
_output_shapes
:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b
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
VariableV2*
shared_name *;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
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
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma
�
AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
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
:���������*
	keep_dims(*

Tidx0
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
A2S/backup_value_network/mulMulA2S/backup_value_network/mul/x2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
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
A2S/backup_value_network/mul_1Mul A2S/backup_value_network/mul_1/xA2S/backup_value_network/Abs*
T0*'
_output_shapes
:���������
�
A2S/backup_value_network/add_1AddA2S/backup_value_network/mulA2S/backup_value_network/mul_1*'
_output_shapes
:���������*
T0
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

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2�*
dtype0*
_output_shapes

:
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
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
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
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
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
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0
�
/A2S/best_value_network/best_value_network/fc0/w
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
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*'
_output_shapes
:���������*
T0
�
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
%A2S/best_value_network/LayerNorm/beta
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
,A2S/best_value_network/LayerNorm/beta/AssignAssign%A2S/best_value_network/LayerNorm/beta7A2S/best_value_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
*A2S/best_value_network/LayerNorm/beta/readIdentity%A2S/best_value_network/LayerNorm/beta*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:*
T0
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
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
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
-A2S/best_value_network/LayerNorm/moments/meanMeanA2S/best_value_network/add?A2S/best_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
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
CA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
1A2S/best_value_network/LayerNorm/moments/varianceMean:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceCA2S/best_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
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
A2S/best_value_network/mul/xConst*
_output_shapes
: *
valueB
 *��?*
dtype0
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
A2S/best_value_network/mul_1/xConst*
_output_shapes
: *
valueB
 *���>*
dtype0
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
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
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
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*'
_output_shapes
:���������*
T0
b
A2S/Reshape/shapeConst*
_output_shapes
:*
valueB"����   *
dtype0
�
A2S/ReshapeReshapeA2S/backup_policy_network/add_2A2S/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
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
	A2S/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
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
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
_output_shapes
: *
T0
�
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*'
_output_shapes
:���������*
T0
\
A2S/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*
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
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
]
A2S/concat/values_0Const*
valueB:*
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

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_2/batch_shape_tensor/BroadcastArgsA2S/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
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
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :������������������*
seed2�*

seed
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :������������������
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*
T0*4
_output_shapes"
 :������������������
h
A2S/addAddA2S/mulA2S/Normal_1/loc*
T0*4
_output_shapes"
 :������������������
h
A2S/Reshape_2/shapeConst*!
valueB"����      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������
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
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  ��
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
VariableV2*
dtype0*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
	container *
shape
:
�
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
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
A2S/backup_q_network/MatMulMatMulA2S/concat_10A2S/backup_q_network/backup_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
VariableV2*
shared_name *7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/LayerNorm/gamma/AssignAssign$A2S/backup_q_network/LayerNorm/gamma5A2S/backup_q_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma
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
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
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
AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
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
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *��̽*
dtype0
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
FA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
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
2A2S/backup_q_network/backup_q_network/out/w/AssignAssign+A2S/backup_q_network/backup_q_network/out/wFA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
validate_shape(*
_output_shapes

:
�
0A2S/backup_q_network/backup_q_network/out/w/readIdentity+A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
�
=A2S/backup_q_network/backup_q_network/out/b/Initializer/zerosConst*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
valueB*    *
dtype0
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
0A2S/backup_q_network/backup_q_network/out/b/readIdentity+A2S/backup_q_network/backup_q_network/out/b*
_output_shapes
:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b
�
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes
: 
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
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
(A2S/best_q_network/LayerNorm/beta/AssignAssign!A2S/best_q_network/LayerNorm/beta3A2S/best_q_network/LayerNorm/beta/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(
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
VariableV2*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
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
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
1A2S/best_q_network/LayerNorm/moments/StopGradientStopGradient)A2S/best_q_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
6A2S/best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
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
,A2S/best_q_network/LayerNorm/batchnorm/add_1Add,A2S/best_q_network/LayerNorm/batchnorm/mul_1*A2S/best_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:���������
]
A2S/best_q_network/mul/xConst*
_output_shapes
: *
valueB
 *��?*
dtype0
�
A2S/best_q_network/mulMulA2S/best_q_network/mul/x,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
}
A2S/best_q_network/AbsAbs,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
_
A2S/best_q_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *���>
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
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
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
A2S/Normal_3/log_prob/subSubA2S/Normal_3/log_prob/mulA2S/Normal_3/log_prob/add*
T0*'
_output_shapes
:���������
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

A2S/Mean_1MeanA2S/advantagesA2S/Const_3*
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
A2S/Mean_1*
T0*
_output_shapes
: 
\
A2S/Const_4Const*
_output_shapes
:*
valueB"       *
dtype0
h

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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

A2S/Mean_3MeanA2S/SquaredDifferenceA2S/Const_5*
	keep_dims( *

Tidx0*
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

A2S/Mean_4MeanA2S/SquaredDifference_1A2S/Const_6*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
A2S/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
A2S/gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
#A2S/gradients/A2S/Mean_2_grad/ShapeShape	A2S/mul_1*
out_type0*
_output_shapes
:*
T0
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
"A2S/gradients/A2S/Mean_2_grad/ProdProd%A2S/gradients/A2S/Mean_2_grad/Shape_1#A2S/gradients/A2S/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
o
%A2S/gradients/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
'A2S/gradients/A2S/Mean_2_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
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
$A2S/gradients/A2S/mul_1_grad/Shape_1ShapeA2S/advantages*
_output_shapes
:*
T0*
out_type0
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
:*
	keep_dims( *

Tidx0*
T0
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
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
:*
	keep_dims( *

Tidx0
�
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
BA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_3/log_prob/Square*'
_output_shapes
:���������*
T0
�
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *   @*
dtype0
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
:*
	keep_dims( *

Tidx0*
T0
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal_1/scale*
T0*'
_output_shapes
:���������
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2*'
_output_shapes
:���������*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
_output_shapes
:*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
:*
	keep_dims( *

Tidx0
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
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
DA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul=^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
�
LA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulE^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul*'
_output_shapes
:���������*
T0
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
4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
:*
	keep_dims( *

Tidx0
�
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
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
IA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*'
_output_shapes
:���������*
T0
y
6A2S/gradients/A2S/best_policy_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1ShapeA2S/best_policy_network/Abs*
_output_shapes
:*
T0*
out_type0
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
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*
N
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
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������*
T0
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
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0*
T0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegNegHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape2A2S/best_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
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
:*
	keep_dims( *

Tidx0*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
SA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill*#
_output_shapes
:���������*
T0*
N
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape2A2S/best_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
: *
	keep_dims( *

Tidx0
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
PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
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
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
out_type0*
_output_shapes
:*
T0
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
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradientN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
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
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
: *
	keep_dims( *

Tidx0
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
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*'
_output_shapes
:���������*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N
�
4A2S/gradients/A2S/best_policy_network/add_grad/ShapeShapeA2S/best_policy_network/MatMul*
_output_shapes
:*
T0*
out_type0
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
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
GA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/add_grad/Reshape@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape*'
_output_shapes
:���������
�
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1*
_output_shapes
:*
T0
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
JA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulC^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
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
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0
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
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
�
AA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    
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
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
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
8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/AssignAssign1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
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
LA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zerosConst*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0
�
:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam
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
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/w/AdamLA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zerosConst*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0
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
LA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zerosConst*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0
�
:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam
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
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(
�
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
�
NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0
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
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/b:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonIA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
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
A2S/gradients_1/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
A2S/gradients_1/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
|
'A2S/gradients_1/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference*
out_type0*
_output_shapes
:*
T0
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
$A2S/gradients_1/A2S/Mean_3_grad/ProdProd'A2S/gradients_1/A2S/Mean_3_grad/Shape_1%A2S/gradients_1/A2S/Mean_3_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
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
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add_2*
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
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
_output_shapes
:*
T0*
out_type0
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
5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
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
5A2S/gradients_1/A2S/best_value_network/add_1_grad/SumSumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
:*
	keep_dims( *

Tidx0
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
3A2S/gradients_1/A2S/best_value_network/mul_grad/mulMulJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_1/A2S/best_value_network/mul_grad/SumSum3A2S/gradients_1/A2S/best_value_network/mul_grad/mulEA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
:*
	keep_dims( *

Tidx0*
T0
�
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1MulA2S/best_value_network/mul_1/xLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
LA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0*
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
:*
	keep_dims( *

Tidx0
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
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*#
_output_shapes
:���������*
T0*
N
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
OA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordivFloorDivLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
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
QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/CastCastQA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truedivRealDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Cast*'
_output_shapes
:���������*
T0
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5A2S/best_value_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
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
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1ProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
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
VariableV2*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:*
dtype0
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
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(
�
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0
�
0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1
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
7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/AssignAssign0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
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
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
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
=A2S/A2S/best_value_network/best_value_network/out/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/w/Adam*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
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
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
�
8A2S/A2S/best_value_network/best_value_network/out/b/Adam
VariableV2*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0
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
LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    
�
:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1
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
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:*
T0
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
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(
�
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
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
%A2S/gradients_2/A2S/Mean_4_grad/ShapeShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
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
$A2S/gradients_2/A2S/Mean_4_grad/ProdProd'A2S/gradients_2/A2S/Mean_4_grad/Shape_1%A2S/gradients_2/A2S/Mean_4_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
'A2S/gradients_2/A2S/Mean_4_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_2/A2S/Mean_4_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
out_type0*
_output_shapes
:*
T0
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
0A2S/gradients_2/A2S/SquaredDifference_1_grad/NegNeg6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_1SumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
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
:*
	keep_dims( *

Tidx0*
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
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
/A2S/gradients_2/A2S/best_q_network/Abs_grad/mulMulHA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_10A2S/gradients_2/A2S/best_q_network/Abs_grad/Sign*'
_output_shapes
:���������*
T0
�
A2S/gradients_2/AddNAddNFA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1/A2S/gradients_2/A2S/best_q_network/Abs_grad/mul*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*
N*'
_output_shapes
:���������*
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/sub*
_output_shapes
:*
T0*
out_type0
�
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:���������
�
A2S/gradients_2/AddN_1AddN\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*'
_output_shapes
:���������*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1
�
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:���������*
T0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1SumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addAdd?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modFloorModFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
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
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/FillFillJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
�
PA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
out_type0*
_output_shapes
:*
T0
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
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradientK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modFloorModBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/MaximumMaximumLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/y*#
_output_shapes
:���������*
T0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*
T0*0
_output_shapes
:������������������*

Tmultiples0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1ProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:���������*
T0
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
:*
	keep_dims( *

Tidx0*
T0
�
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
FA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
_output_shapes
:*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1
�
5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
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
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
�
A2S/beta2_power_2/readIdentityA2S/beta2_power_2*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: *
T0
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
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
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
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
�
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/b/AdamBA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
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
1A2S/A2S/best_q_network/LayerNorm/beta/Adam/AssignAssign*A2S/A2S/best_q_network/LayerNorm/beta/Adam<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zeros*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
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
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
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
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
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
VariableV2*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
	container *
shape:*
dtype0
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
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
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
A2S/Adam_2/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
W
A2S/Adam_2/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/b0A2S/A2S/best_q_network/best_q_network/out/b/Adam2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonHA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
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
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: *
T0
�
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
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
T0"rV���     (+�E	%�M?�z�AJ��
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
shape:*
dtype0*
_output_shapes
:
Z
A2S/stddev_policy_oldPlaceholder*
dtype0*
_output_shapes
:*
shape:
n
A2S/returnsPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
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
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  �?
�
^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
seed2*
dtype0*
_output_shapes

:
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
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
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
:A2S/backup_policy_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
valueB*    
�
(A2S/backup_policy_network/LayerNorm/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta
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
:���������*
	keep_dims(*

Tidx0*
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
:���������*
	keep_dims(*

Tidx0
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
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
	container 
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
	container *
shape:
�
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
�
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
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
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:*
T0
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
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
,A2S/best_policy_network/LayerNorm/gamma/readIdentity'A2S/best_policy_network/LayerNorm/gamma*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
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
:���������*
	keep_dims(*

Tidx0
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
:���������*
	keep_dims(*

Tidx0*
T0
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
1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtRsqrt/A2S/best_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:���������
�
/A2S/best_policy_network/LayerNorm/batchnorm/mulMul1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt,A2S/best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
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
/A2S/best_policy_network/LayerNorm/batchnorm/subSub+A2S/best_policy_network/LayerNorm/beta/read1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:���������*
T0
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
A2S/best_policy_network/AbsAbs1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
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

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2u*
dtype0*
_output_shapes

:
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
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/add_16A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(*
_output_shapes
:
�
8A2S/backup_value_network/backup_value_network/fc0/b/readIdentity3A2S/backup_value_network/backup_value_network/fc0/b*
_output_shapes
:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b
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
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
	container 
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
,A2S/backup_value_network/LayerNorm/beta/readIdentity'A2S/backup_value_network/LayerNorm/beta*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta
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
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
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
3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
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
2A2S/backup_value_network/LayerNorm/batchnorm/mul_2Mul/A2S/backup_value_network/LayerNorm/moments/mean0A2S/backup_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
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
A2S/backup_value_network/mul/xConst*
_output_shapes
: *
valueB
 *��?*
dtype0
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
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2�*
dtype0
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: *
T0
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
�
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
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
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
8A2S/backup_value_network/backup_value_network/out/b/readIdentity3A2S/backup_value_network/backup_value_network/out/b*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
_output_shapes
:*
T0
�
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_value_network/add_2Add!A2S/backup_value_network/MatMul_18A2S/backup_value_network/backup_value_network/out/b/read*'
_output_shapes
:���������*
T0
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
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(
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
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
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
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
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
-A2S/best_value_network/LayerNorm/moments/meanMeanA2S/best_value_network/add?A2S/best_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
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
1A2S/best_value_network/LayerNorm/moments/varianceMean:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceCA2S/best_value_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
u
0A2S/best_value_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
�
.A2S/best_value_network/LayerNorm/batchnorm/addAdd1A2S/best_value_network/LayerNorm/moments/variance0A2S/best_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
�
0A2S/best_value_network/LayerNorm/batchnorm/RsqrtRsqrt.A2S/best_value_network/LayerNorm/batchnorm/add*'
_output_shapes
:���������*
T0
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
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"      
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
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
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
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:*
T0
�
AA2S/best_value_network/best_value_network/out/b/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0
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
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0
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
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
_output_shapes
: *
T0
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
A2S/Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
�
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
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
'A2S/Normal_2/batch_shape_tensor/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
]
A2S/concat/values_0Const*
valueB:*
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
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :������������������*
seed2�*

seed
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :������������������
�
A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*4
_output_shapes"
 :������������������*
T0
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*4
_output_shapes"
 :������������������*
T0
h
A2S/addAddA2S/mulA2S/Normal_1/loc*4
_output_shapes"
 :������������������*
T0
h
A2S/Reshape_2/shapeConst*!
valueB"����      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������
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
LA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB"      *
dtype0
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
FA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:*
T0
�
+A2S/backup_q_network/backup_q_network/fc0/w
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
	container *
shape
:*
dtype0*
_output_shapes

:
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
5A2S/backup_q_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
valueB*    
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
(A2S/backup_q_network/LayerNorm/beta/readIdentity#A2S/backup_q_network/LayerNorm/beta*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
_output_shapes
:*
T0
�
5A2S/backup_q_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
valueB*  �?
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
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:*
T0
�
=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
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
:���������*
	keep_dims(*

Tidx0*
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
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
seed2�
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes
: *
T0
�
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
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
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
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
:*
dtype0*
_output_shapes

:*
shared_name 
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
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*
_output_shapes
:*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  �?*
dtype0
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
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
1A2S/best_q_network/LayerNorm/moments/StopGradientStopGradient)A2S/best_q_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
6A2S/best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
q
,A2S/best_q_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
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
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
�
'A2S/best_q_network/best_q_network/out/w
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
A2S/Normal_3/log_prob/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *   �
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

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*
	keep_dims( *

Tidx0*
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

A2S/Mean_3MeanA2S/SquaredDifferenceA2S/Const_5*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
r
A2S/value_network_loss/tagsConst*
dtype0*
_output_shapes
: *'
valueB BA2S/value_network_loss
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

A2S/Mean_4MeanA2S/SquaredDifference_1A2S/Const_6*
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
%A2S/gradients/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_2_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
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
%A2S/gradients/A2S/Mean_2_grad/Shape_1Shape	A2S/mul_1*
_output_shapes
:*
T0*
out_type0
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
: *
	keep_dims( *

Tidx0*
T0
o
%A2S/gradients/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
�
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
:*
	keep_dims( *

Tidx0*
T0
�
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
GA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
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
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1A2S/Normal_1/scale*'
_output_shapes
:���������*
T0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
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
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*
T0*'
_output_shapes
:���������
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
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal_1/loc*
T0*
out_type0*
_output_shapes
:
�
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
_output_shapes
:*
T0
�
BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
IA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1
�
QA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*S
_classI
GEloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:���������*
T0
�
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
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
:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
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
4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
:*
	keep_dims( *

Tidx0
�
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1MulA2S/best_policy_network/mul/xIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_1Sum4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1FA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
3A2S/gradients/A2S/best_policy_network/Abs_grad/SignSign1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
�
2A2S/gradients/A2S/best_policy_network/Abs_grad/mulMulKA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_13A2S/gradients/A2S/best_policy_network/Abs_grad/Sign*'
_output_shapes
:���������*
T0
�
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*
N
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_policy_network/add]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtA2S/gradients/AddN_1*'
_output_shapes
:���������*
T0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape2A2S/best_policy_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
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
:*
	keep_dims( *

Tidx0*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape2A2S/best_policy_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
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
: *
	keep_dims( *

Tidx0
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
: *
	keep_dims( *

Tidx0
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
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0
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
UA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulUA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
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
:*
	keep_dims( *

Tidx0*
T0
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
:*
	keep_dims( *

Tidx0
�
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
_A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpW^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeS^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
gA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*i
_class_
][loc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������*
T0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
N*#
_output_shapes
:���������*
T0
�
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*
	keep_dims( *

Tidx0*
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*
	keep_dims( *

Tidx0*
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
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
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
DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1*
_output_shapes
:
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
JA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulC^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul*'
_output_shapes
:���������
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
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape: 
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
LA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zerosConst*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0
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
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/w/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
�
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0
�
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1
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
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
�
4A2S/A2S/best_policy_network/LayerNorm/beta/Adam/readIdentity/A2S/A2S/best_policy_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
�
CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    
�
1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1
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
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(
�
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0
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
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zerosConst*
_output_shapes

:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0
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
CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
�
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
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
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(
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
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:*
T0
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
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/w:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( 
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
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: *
T0
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
A2S/gradients_1/FillFillA2S/gradients_1/ShapeA2S/gradients_1/Const*
_output_shapes
: *
T0
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
%A2S/gradients_1/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
$A2S/gradients_1/A2S/Mean_3_grad/ProdProd'A2S/gradients_1/A2S/Mean_3_grad/Shape_1%A2S/gradients_1/A2S/Mean_3_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
k
)A2S/gradients_1/A2S/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
'A2S/gradients_1/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_3_grad/Prod_1)A2S/gradients_1/A2S/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
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
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
:*
	keep_dims( *

Tidx0
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
BA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
�
JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape*'
_output_shapes
:���������*
T0
�
LA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
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
9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1ShapeA2S/best_value_network/mul_1*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5A2S/gradients_1/A2S/best_value_network/add_1_grad/SumSumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_1SumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
3A2S/gradients_1/A2S/best_value_network/mul_grad/SumSum3A2S/gradients_1/A2S/best_value_network/mul_grad/mulEA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/mul_grad/Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1MulA2S/best_value_network/mul/xJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1MulA2S/best_value_network/mul_1/xLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
OA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordivFloorDivLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
: *
	keep_dims( *

Tidx0
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
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truedivRealDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:���������*
T0
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
7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_17A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
@A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1
�
HA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape
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
;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
CA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul<^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1
�
KA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulD^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
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
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
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
dtype0*
_output_shapes
: *
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape: 
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
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: *
T0
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
AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/b/AdamJA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
=A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:*
T0
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
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
AA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0
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
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
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
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
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
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
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
VariableV2*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0
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
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(
�
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
U
A2S/Adam_1/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/b8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonJA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
use_nesterov( *
_output_shapes
:
�
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( 
�
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
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
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( 
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
%A2S/gradients_2/A2S/Mean_4_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
$A2S/gradients_2/A2S/Mean_4_grad/ProdProd'A2S/gradients_2/A2S/Mean_4_grad/Shape_1%A2S/gradients_2/A2S/Mean_4_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'A2S/gradients_2/A2S/Mean_4_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
HA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1*
_output_shapes
:
�
7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
/A2S/gradients_2/A2S/best_q_network/mul_grad/SumSum/A2S/gradients_2/A2S/best_q_network/mul_grad/mulAA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
FA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*'
_output_shapes
:���������
v
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
/A2S/gradients_2/A2S/best_q_network/Abs_grad/mulMulHA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_10A2S/gradients_2/A2S/best_q_network/Abs_grad/Sign*'
_output_shapes
:���������*
T0
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
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul)A2S/best_q_network/LayerNorm/moments/meanZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0
�
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/FillFillJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
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
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordivFloorDivHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
out_type0*
_output_shapes
:*
T0
�
aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeSA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
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
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1cA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeConst*
_output_shapes
: *
value	B :*
dtype0
�
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addAdd;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
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
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*
T0*0
_output_shapes
:������������������*

Tmultiples0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
�
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
: *
	keep_dims( *

Tidx0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/CastCastIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:���������*
T0
�
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������*
T0
�
1A2S/gradients_2/A2S/best_q_network/add_grad/ShapeShapeA2S/best_q_network/MatMul*
_output_shapes
:*
T0*
out_type0
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
/A2S/gradients_2/A2S/best_q_network/add_grad/SumSumA2S/gradients_2/AddN_2AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
?A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul8^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
�
GA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
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
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
_output_shapes
: *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
�
BA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zerosConst*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0
�
0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
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
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
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
VariableV2*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:*
dtype0
�
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/b/AdamBA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
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
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/AssignAssign+A2S/A2S/best_q_network/LayerNorm/gamma/Adam=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zeros*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
BA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
�
0A2S/A2S/best_q_network/best_q_network/out/w/Adam
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
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
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
7A2S/A2S/best_q_network/best_q_network/out/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/b/AdamBA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(
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
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
�
7A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
U
A2S/Adam_2/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/b0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonFA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
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
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
use_nesterov( 
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
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: *
T0
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
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(
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
T0""�b
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
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0"�,
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
)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0"�
	summaries�
�
A2S/kl:0
A2S/average_advantage:0
A2S/policy_network_loss:0
A2S/value_network_loss:0
A2S/q_network_loss:0
A2S/average_reward_1:0"0
train_op$
"
A2S/Adam

A2S/Adam_1

A2S/Adam_2���*       ����	�z?�z�A*

A2S/average_reward_1  0A��6*       ����	m�~?�z�A*

A2S/average_reward_1  (AB�P�*       ����	�y�?�z�A*

A2S/average_reward_1UUA�3q*       ����	��?�z�A**

A2S/average_reward_1  (A
�n*       ����	�M�?�z�A3*

A2S/average_reward_133#A�g}%*       ����	��?�z�A@*

A2S/average_reward_1��*A(| �*       ����	�?�z�AH*

A2S/average_reward_1I�$A�t��*       ����	�n�?�z�AS*

A2S/average_reward_1  &A�#�*       ����	�$�?�z�A`*

A2S/average_reward_1��*A�-�B*       ����	Xl�?�z�Aj*

A2S/average_reward_1��)A�@�]*       ����	)j�?�z�Aq*

A2S/average_reward_1]$A+��*       ����	���?�z�Az*

A2S/average_reward_1��"A�w�K+       ��K	=4�?�z�A�*

A2S/average_reward_1��A�O�+       ��K	�Ĵ?�z�A�*

A2S/average_reward_1I�$A�1��+       ��K	@�?�z�A�*

A2S/average_reward_1DD$A���+       ��K	�2�?�z�A�*

A2S/average_reward_1  #A�i�+       ��K	��?�z�A�*

A2S/average_reward_1��%ABY�+       ��K	&y�?�z�A�*

A2S/average_reward_1��"Ax2��+       ��K	�H�?�z�A�*

A2S/average_reward_1�� AT<��+       ��K	.Y�?�z�A�*

A2S/average_reward_1   A���+       ��K	���?�z�A�*

A2S/average_reward_1�!AiMj�+       ��K	�?�z�A�*

A2S/average_reward_1]t!A[�b+       ��K	�4�?�z�A�*

A2S/average_reward_1� A��U�+       ��K	��?�z�A�*

A2S/average_reward_1UU!A-&��+       ��K	���?�z�A�*

A2S/average_reward_1�G!Aw�G+       ��K	D��?�z�A�*

A2S/average_reward_1�� A�
bt+       ��K	d��?�z�A�*

A2S/average_reward_1��A^=�
+       ��K	r?�?�z�A�*

A2S/average_reward_1   A�z��+       ��K	F�?�z�A�*

A2S/average_reward_1��AlO�f+       ��K	ߙ@�z�A�*

A2S/average_reward_1��A��h/�       L	vM	�8L@�z�A�*�

A2S/klN��<

A2S/average_advantage�b�

A2S/policy_network_losso༾

A2S/value_network_lossܰh>

A2S/q_network_lossL�/>���-+       ��K	˙T@�z�A�*

A2S/average_reward_1�9'A�MO+       ��K	��d@�z�A�*

A2S/average_reward_1 �2A���,+       ��K	G�}@�z�A�*

A2S/average_reward_1��CA�>j+       ��K	b�@�z�A�*

A2S/average_reward_1--EA�<�#+       ��K	�a�@�z�A�*

A2S/average_reward_1��KA8f�+       ��K	�ť@�z�A�*

A2S/average_reward_1�YA9�pf+       ��K	���@�z�A�*

A2S/average_reward_10EnA9^Y�+       ��K	�B�@�z�A�*

A2S/average_reward_1�PvA�G� +       ��K	>��@�z�A�*

A2S/average_reward_1\��A_gT+       ��K	�
A�z�A�*

A2S/average_reward_1ff�A���+       ��K	��A�z�A�*

A2S/average_reward_1%j�A]<+       ��K	��7A�z�A�*

A2S/average_reward_1�a�A�B�+       ��K	��SA�z�A�*

A2S/average_reward_1_ЗA�΁�+       ��K	�?ZA�z�A�*

A2S/average_reward_1�E�A\t*I+       ��K	�3bA�z�A�*

A2S/average_reward_1r�A�FE+       ��K	��|A�z�A�*

A2S/average_reward_1zӛA+?�+       ��K	�یA�z�A�*

A2S/average_reward_1<�A���+       ��K	h{�A�z�A�*

A2S/average_reward_1  �A��<+       ��K	Z=�A�z�A�*

A2S/average_reward_1��As��n+       ��K	�q�A�z�A�*

A2S/average_reward_1�G�Aͳ�+       ��K	:n�A�z�A�*

A2S/average_reward_1���A��+       ��K	uk�A�z�A�*

A2S/average_reward_1vb�A�H�+       ��K	h�B�z�A�	*

A2S/average_reward_1'�A��P{+       ��K	��B�z�A�	*

A2S/average_reward_1���A�M3+       ��K	�'B�z�A�	*

A2S/average_reward_1X��A�	�+       ��K	��2B�z�A�	*

A2S/average_reward_1�m�A���2+       ��K	M�IB�z�A�	*

A2S/average_reward_1���A~��S+       ��K	ոSB�z�A�	*

A2S/average_reward_1O#�A\Q�>+       ��K	�Z^B�z�A�
*

A2S/average_reward_1ݗ�A�<��+       ��K	��nB�z�A�
*

A2S/average_reward_1���A�"4��       L	vM	�=�B�z�A�
*�

A2S/kl`��>

A2S/average_advantage�Y��

A2S/policy_network_loss�O�

A2S/value_network_loss#��A

A2S/q_network_lossTB1�{�+       ��K	��B�z�A�
*

A2S/average_reward_1L��A�V+       ��K	@g�B�z�A�
*

A2S/average_reward_1眯A�|�+       ��K	���B�z�A�
*

A2S/average_reward_1nۮAG���+       ��K	q��B�z�A�
*

A2S/average_reward_1 ��Avz�h+       ��K	�,�B�z�A�*

A2S/average_reward_1�^�A9Ⱦq+       ��K		��B�z�A�*

A2S/average_reward_1UU�A��x�+       ��K	��B�z�A�*

A2S/average_reward_1�A���+       ��K	RC�z�A�*

A2S/average_reward_1--�A�Ӓ�+       ��K	b�C�z�A�*

A2S/average_reward_1�3�A����+       ��K	�O C�z�A�*

A2S/average_reward_1_�A;])%+       ��K	(*C�z�A�*

A2S/average_reward_1�ͱA�!HM+       ��K	h$4C�z�A�*

A2S/average_reward_1�q�A+!��+       ��K	��<C�z�A�*

A2S/average_reward_1bİA� ��+       ��K	�QC�z�A�*

A2S/average_reward_1�|�A�@�+       ��K	�sXC�z�A�*

A2S/average_reward_1贱AND�+       ��K	o�aC�z�A�*

A2S/average_reward_1Cy�A�:+       ��K	��mC�z�A�*

A2S/average_reward_1Z��A�ǿ�+       ��K	�vC�z�A�*

A2S/average_reward_1�o�AM���+       ��K	�h~C�z�A�*

A2S/average_reward_1x��A ��,+       ��K	i��C�z�A�*

A2S/average_reward_1�̱A,��I+       ��K	15�C�z�A�*

A2S/average_reward_1��A���w+       ��K	�ܾC�z�A�*

A2S/average_reward_1���Au�+       ��K	?n�C�z�A�*

A2S/average_reward_1u~�A�b�+       ��K	y��C�z�A�*

A2S/average_reward_1�0�A��+       ��K	o��C�z�A�*

A2S/average_reward_1xx�A��\+       ��K	��C�z�A�*

A2S/average_reward_10�A+��+       ��K	��D�z�A�*

A2S/average_reward_1R��AN�=+       ��K	��#D�z�A�*

A2S/average_reward_1]t�A��Z�+       ��K	-�4D�z�A�*

A2S/average_reward_1?��A-��+       ��K	�@D�z�A�*

A2S/average_reward_1���A��˿�       L	vM		+�D�z�A�*�

A2S/kl٧w>

A2S/average_advantageC^�

A2S/policy_network_loss����

A2S/value_network_losst�FA

A2S/q_network_loss��A�Ss�+       ��K	�p�D�z�A�*

A2S/average_reward_1˨�A�R�+       ��K	�D�z�A�*

A2S/average_reward_1YȾAI��-+       ��K	x��D�z�A�*

A2S/average_reward_1��AN̐�+       ��K	'R�D�z�A�*

A2S/average_reward_1�w�A^�t+       ��K	�zE�z�A�*

A2S/average_reward_1F��A=�O+       ��K	��E�z�A�*

A2S/average_reward_1UU�AU��t+       ��K	��E�z�A�*

A2S/average_reward_1��A��3O+       ��K	/D�E�z�A�*

A2S/average_reward_1N��An�}.+       ��K	I./F�z�A�*

A2S/average_reward_1K��AU�{+       ��K	�cfF�z�A�*

A2S/average_reward_1��A�A�3+       ��K	GO�F�z�A�*

A2S/average_reward_1R� B���+       ��K	��F�z�A�*

A2S/average_reward_1)\B˫�E+       ��K	j_ G�z�A�*

A2S/average_reward_1=
B_l+       ��K	�pG�z�A�*

A2S/average_reward_1  B��Z�+       ��K	��1G�z�A�*

A2S/average_reward_1��B4T�+       ��K	kKfG�z�A�*

A2S/average_reward_1q=
B���j+       ��K	�ɦG�z�A�*

A2S/average_reward_1�zB��N�+       ��K	g��G�z�A�*

A2S/average_reward_133BtlL+       ��K	���G�z�A�*

A2S/average_reward_1ףB��k�+       ��K	(+�G�z�A�*

A2S/average_reward_1=
Bi�#�+       ��K	�H�z�A�*

A2S/average_reward_1��B��́+       ��K	��-H�z�A�*

A2S/average_reward_1��B����+       ��K	W�GH�z�A�*

A2S/average_reward_1R�B�oU+       ��K	�~_H�z�A�*

A2S/average_reward_1=
B��+       ��K	��H�z�A� *

A2S/average_reward_1�G B��S +       ��K	�Y�H�z�A� *

A2S/average_reward_1{"B��+       ��K	)+I�z�A�"*

A2S/average_reward_1��)B�[+       ��K	aŮI�z�A�$*

A2S/average_reward_1ף2B^�L+       ��K	5Q�I�z�A�%*

A2S/average_reward_1=
6B-�Q�+       ��K	��J�z�A�%*

A2S/average_reward_1q=9B8����       L	vM	@6XJ�z�A�%*�

A2S/kl���>

A2S/average_advantage0%<

A2S/policy_network_loss��>

A2S/value_network_loss&6�B

A2S/q_network_lossk��B`�+       ��K	ިhJ�z�A�&*

A2S/average_reward_1�Q:B3�3^+       ��K	V�rJ�z�A�&*

A2S/average_reward_1��:B�c�1+       ��K	pJ{J�z�A�&*

A2S/average_reward_1  ;Bg�!�+       ��K	ar�J�z�A�&*

A2S/average_reward_1��;B��p+       ��K	��J�z�A�&*

A2S/average_reward_1��<B�7ϩ+       ��K	�J�z�A�&*

A2S/average_reward_1�=B��b+       ��K	��J�z�A�'*

A2S/average_reward_1�>B.�>Q+       ��K	�J�z�A�'*

A2S/average_reward_1ף>B���+       ��K	|��J�z�A�'*

A2S/average_reward_1
�>B���+       ��K	m�J�z�A�'*

A2S/average_reward_1  ?B��^�+       ��K	��J�z�A�'*

A2S/average_reward_1�?B7Nu�+       ��K	���J�z�A�(*

A2S/average_reward_1R�>B�#�%+       ��K	���J�z�A�(*

A2S/average_reward_1�=B�|;K+       ��K	�E K�z�A�(*

A2S/average_reward_1�Q=B�K*+       ��K	V�K�z�A�(*

A2S/average_reward_1ף<B,	y)+       ��K	'
K�z�A�(*

A2S/average_reward_1�Q;B��R�+       ��K	��K�z�A�(*

A2S/average_reward_1�9Bp�!�+       ��K	# K�z�A�(*

A2S/average_reward_1
�8Bx�]�+       ��K	��%K�z�A�(*

A2S/average_reward_1
�6B�5+       ��K	�B,K�z�A�(*

A2S/average_reward_1�G6BlK�B+       ��K	>�KK�z�A�)*

A2S/average_reward_1R�7Bȩ�7+       ��K	�@SK�z�A�)*

A2S/average_reward_1ף5B���+       ��K	Q�ZK�z�A�)*

A2S/average_reward_1��3B���+       ��K	[�_K�z�A�)*

A2S/average_reward_1�z3B!���+       ��K	�eK�z�A�)*

A2S/average_reward_1�(3B���+       ��K	?�lK�z�A�)*

A2S/average_reward_1��1B�]�+       ��K	�/tK�z�A�)*

A2S/average_reward_1331BP���+       ��K	�'~K�z�A�)*

A2S/average_reward_1{1B���s+       ��K	�K�z�A�**

A2S/average_reward_1�.B����+       ��K	���K�z�A�**

A2S/average_reward_1��.B�j-ۖ       L	vM	���K�z�A�**�

A2S/kl)�B>

A2S/average_advantage�5�

A2S/policy_network_lossFaA�

A2S/value_network_loss���A

A2S/q_network_loss���A3OM�+       ��K	"J�K�z�A�**

A2S/average_reward_1��.B��̣+       ��K	[�K�z�A�**

A2S/average_reward_1��,B���L+       ��K	~��K�z�A�**

A2S/average_reward_1�+B���+       ��K	PI�K�z�A�**

A2S/average_reward_1q=,B �O+       ��K	��L�z�A�**

A2S/average_reward_1��+B���w+       ��K	��L�z�A�+*

A2S/average_reward_1��+B�F��+       ��K	\ #L�z�A�+*

A2S/average_reward_1R�+B���+       ��K	�6L�z�A�+*

A2S/average_reward_1ff,BM�/�+       ��K	�XAL�z�A�+*

A2S/average_reward_1\�,B���+       ��K	��EL�z�A�+*

A2S/average_reward_1
�+BA�+       ��K	�LL�z�A�,*

A2S/average_reward_1�+B���+       ��K	ρQL�z�A�,*

A2S/average_reward_1ff+B�@�+       ��K	=QYL�z�A�,*

A2S/average_reward_1�z+B!� +       ��K	8�dL�z�A�,*

A2S/average_reward_1�p+B���+       ��K	�1nL�z�A�,*

A2S/average_reward_1{+BK4�+       ��K	��tL�z�A�,*

A2S/average_reward_1)\*B�_�+       ��K	�L�z�A�,*

A2S/average_reward_1��*B9�:+       ��K	���L�z�A�-*

A2S/average_reward_1�*Bԫ��+       ��K	�L�z�A�-*

A2S/average_reward_1��)B�+       ��K	&'�L�z�A�-*

A2S/average_reward_1)\)Bro!�+       ��K	# �L�z�A�-*

A2S/average_reward_1ff)B�j�+       ��K		զL�z�A�-*

A2S/average_reward_1�Q)B˅�+       ��K	��L�z�A�-*

A2S/average_reward_1  *B�/��+       ��K	��L�z�A�.*

A2S/average_reward_1�)B�c�+       ��K	^�L�z�A�.*

A2S/average_reward_1�)B���+       ��K	Z��L�z�A�.*

A2S/average_reward_1��)B��b�+       ��K	��L�z�A�.*

A2S/average_reward_1��)B��w�+       ��K	d1�L�z�A�.*

A2S/average_reward_1��)Bǫb�+       ��K	�x�L�z�A�.*

A2S/average_reward_1R�)B��Z'+       ��K	fz�L�z�A�.*

A2S/average_reward_1�Q)B��Y+       ��K	�M�z�A�/*

A2S/average_reward_1��'B�L�+       ��K	�M�z�A�/*

A2S/average_reward_1�z'B�(K�+       ��K	�YM�z�A�/*

A2S/average_reward_1��'B��?+       ��K	XiM�z�A�/*

A2S/average_reward_1{&B����+       ��K	�# M�z�A�/*

A2S/average_reward_1��%B6m��+       ��K	+M�z�A�/*

A2S/average_reward_1{&B��+       ��K	`q0M�z�A�/*

A2S/average_reward_1�G%Bf@+       ��K	��@M�z�A�0*

A2S/average_reward_1)\%BJ��D+       ��K	}JM�z�A�0*

A2S/average_reward_1��$B|��+       ��K	5�PM�z�A�0*

A2S/average_reward_1\�$B`�c�       L	vM	H�M�z�A�0*�

A2S/kl>F?

A2S/average_advantage_D�>

A2S/policy_network_lossH-�>

A2S/value_network_loss`r�?

A2S/q_network_loss�Ș?Z���+       ��K	�aN�z�A�3*

A2S/average_reward_1�3B��x+       ��K	��UO�z�A�7*

A2S/average_reward_1�zEBCq��+       ��K	�xpP�z�A�<*

A2S/average_reward_1�(ZB�9Y�+       ��K	^�Q�z�A�@*

A2S/average_reward_1��lB䌎A+       ��K	��hR�z�A�D*

A2S/average_reward_1=
uBtf	�+       ��K	��FS�z�A�G*

A2S/average_reward_1��B<8܂+       ��K	x�JT�z�A�L*

A2S/average_reward_1�z�B���+       ��K	?�.U�z�A�O*

A2S/average_reward_1��B?��+       ��K	#�*V�z�A�S*

A2S/average_reward_1�u�Bp�+       ��K	�l�V�z�A�W*

A2S/average_reward_1H�Bi�J�+       ��K	L��W�z�A�[*

A2S/average_reward_1=��B-��+       ��K	��X�z�A�^*

A2S/average_reward_1q��B�h��+       ��K	"I�Y�z�A�b*

A2S/average_reward_1�p�B@5��+       ��K	}��Z�z�A�e*

A2S/average_reward_1�k�B�%��+       ��K	���[�z�A�j*

A2S/average_reward_1)��B�h�+       ��K	���\�z�A�m*

A2S/average_reward_1���B�e�+       ��K	#q�]�z�A�q*

A2S/average_reward_1��B#��+       ��K	/<|^�z�A�t*

A2S/average_reward_1R8�B 9&�+       ��K	||�_�z�A�x*

A2S/average_reward_1���BX�a�+       ��K	P�`�z�A�|*

A2S/average_reward_13��BW��,       ���E	���a�z�AՀ*

A2S/average_reward_1�#�B�^,       ���E	��^b�z�AЃ*

A2S/average_reward_1�QC���,       ���E	h�Lc�z�A��*

A2S/average_reward_1��C���,       ���E	P�d�z�A��*

A2S/average_reward_1�pCsM8�,       ���E	���d�z�A��*

A2S/average_reward_1�:C6%��,       ���E	pl�e�z�A��*

A2S/average_reward_1f�Cl㼹,       ���E	���f�z�A��*

A2S/average_reward_1��C9��,       ���E	�@Eg�z�A��*

A2S/average_reward_1�LChndB,       ���E	��&h�z�A�*

A2S/average_reward_1��C�j�,       ���E	�B�h�z�A��*

A2S/average_reward_1�(C߳!�,       ���E	���i�z�A��*

A2S/average_reward_1õC��^!,       ���E	��j�z�A��*

A2S/average_reward_1=�#C�:e,       ���E	���k�z�A��*

A2S/average_reward_1�(C�p�,       ���E	�Q�l�z�A��*

A2S/average_reward_1 @-C^���,       ���E	��m�z�A��*

A2S/average_reward_1  1C�-y,       ���E	x��n�z�A��*

A2S/average_reward_1�:6C�#J',       ���E	�'�o�z�Aι*

A2S/average_reward_1�:;C�s��,       ���E	V��p�z�A��*

A2S/average_reward_1E?Cݔ; ,       ���E	�w�q�z�A��*

A2S/average_reward_1q�CCȼ�O,       ���E	Zr�z�A��*

A2S/average_reward_1{�GC�#��,       ���E	��Os�z�A��*

A2S/average_reward_1�BLCb��,       ���E	�G=t�z�A��*

A2S/average_reward_1��PC�KC,       ���E	�$u�z�A��*

A2S/average_reward_1R�TC�},       ���E	E�v�z�A��*

A2S/average_reward_1�QYC���,       ���E	�Ew�z�A��*

A2S/average_reward_1�^C�n2!,       ���E	�[x�z�A��*

A2S/average_reward_1��cC��;�,       ���E	���x�z�A��*

A2S/average_reward_1
WhC�F�*,       ���E	'��y�z�A��*

A2S/average_reward_1H!lC
rq�,       ���E	��z�z�A��*

A2S/average_reward_13spC�?�,       ���E	3�}{�z�A��*

A2S/average_reward_1q}tC���D�       �v@�	���{�z�A��*�

A2S/kl��>

A2S/average_advantage>�>

A2S/policy_network_lossՉ�=

A2S/value_network_losse��A

A2S/q_network_lossF�A���,       ���E	8��{�z�A��*

A2S/average_reward_1�tC���,       ���E	B|�{�z�A��*

A2S/average_reward_1��tC����,       ���E	�|�z�A��*

A2S/average_reward_1{TuC�%�,       ���E	�(|�z�A��*

A2S/average_reward_1õuC=+l,       ���E	��~�z�A��*

A2S/average_reward_1)�C;���,       ���E	��7~�z�A��*

A2S/average_reward_1q�C ��,       ���E	�XJ~�z�A��*

A2S/average_reward_1
�C쎐,       ���E	<�k~�z�A��*

A2S/average_reward_1�Q�Cp	6,       ���E	��a��z�A��*

A2S/average_reward_1�B�CH�,       ���E	��0��z�A��*

A2S/average_reward_1�,�C
�`0,       ���E	#nN��z�A��*

A2S/average_reward_1
W�Cgd��,       ���E	�G��z�A��*

A2S/average_reward_1HA�C]�1�,       ���E	~+f��z�A�*

A2S/average_reward_1f��C7�,       ���E	��|��z�A��*

A2S/average_reward_1{��C�z^�,       ���E	wv��z�A��*

A2S/average_reward_1�CK�O,       ���E	R���z�A��*

A2S/average_reward_13ӘC�f��,       ���E	"���z�A��*

A2S/average_reward_1��COCtK,       ���E	$���z�A��*

A2S/average_reward_1 ��C���,       ���E	VA���z�A��*

A2S/average_reward_1H��C���,       ���E	�؝��z�A�*

A2S/average_reward_1H�C�W�J,       ���E	�2���z�A��*

A2S/average_reward_1
�C:=��,       ���E	ޤȍ�z�A��*

A2S/average_reward_1��Ce>W�,       ���E	oz���z�A��*

A2S/average_reward_1�"�C��(�,       ���E	�ݏ�z�A��*

A2S/average_reward_1H�CW�7�,       ���E	���z�A��*

A2S/average_reward_1
�C��I,       ���E	�0��z�A�*

A2S/average_reward_1q=�C@��,       ���E	ލ��z�A��*

A2S/average_reward_1=J�C�۪,       ���E	��$��z�A��*

A2S/average_reward_1�c�C��D:,       ���E	���z�A��*

A2S/average_reward_13S�C���,       ���E	� %��z�A��*

A2S/average_reward_13s�C��,�,       ���E	$��z�A��*

A2S/average_reward_1
W�C�Ɠ�,       ���E	'{��z�A��*

A2S/average_reward_1HA�C��6G,       ���E	S��z�A��*

A2S/average_reward_1\O�CH��B,       ���E	x�+��z�A��*

A2S/average_reward_1.�C9A,       ���E	�>��z�A��*

A2S/average_reward_1\O�C�!�,       ���E	�rY��z�A��*

A2S/average_reward_1{t�C����,       ���E	3ZQ��z�A��*

A2S/average_reward_1�L�C'1��,       ���E	����z�A��*

A2S/average_reward_1���C���,       ���E	�����z�A��*

A2S/average_reward_1�p�Co�$c,       ���E	�s���z�A��*

A2S/average_reward_13S�C�/],       ���E	g�͟�z�A��*

A2S/average_reward_1\��C�o�,       ���E	�����z�A��*

A2S/average_reward_1H��Ca��,       ���E	wzˡ�z�A��*

A2S/average_reward_1���C�ԟ�,       ���E	D��z�Aʀ*

A2S/average_reward_13��Cb�Ly,       ���E	����z�A��*

A2S/average_reward_1
��C[���,       ���E	i��z�A��*

A2S/average_reward_1��C�/�,       ���E	A})��z�A�*

A2S/average_reward_1\/�C���,       ���E	�37��z�A��*

A2S/average_reward_1.�Cr��e,       ���E	O-+��z�A�*

A2S/average_reward_1��C�\��,       ���E	�Oe��z�AЊ*

A2S/average_reward_1���C~����       �v@�	*����z�AЊ*�

A2S/kl��>

A2S/average_advantage���?

A2S/policy_network_lossGY�>

A2S/value_network_loss`��C

A2S/q_network_loss�k�Cq8�,       ���E	�δ��z�A܊*

A2S/average_reward_1n�C��	,       ���E	�����z�A�*

A2S/average_reward_1=
�Cp�K�,       ���E	�ä�z�A��*

A2S/average_reward_1
7�C��,       ���E	��̤�z�A��*

A2S/average_reward_13��Cx��6,       ���E	*�Ӧ�z�A�*

A2S/average_reward_1E�Ce6;�,       ���E	f�ܦ�z�A��*

A2S/average_reward_1���C��},       ���E	���z�A��*

A2S/average_reward_1
7�C���[,       ���E	����z�A��*

A2S/average_reward_1R��C�
݀,       ���E	����z�A��*

A2S/average_reward_1��C��Ϟ,       ���E	h����z�A��*

A2S/average_reward_1��CT��,       ���E	M���z�Aؓ*

A2S/average_reward_1{�C�l�,       ���E	z{��z�A�*

A2S/average_reward_13��C6Xٸ,       ���E	9���z�A�*

A2S/average_reward_1
��C�b��,       ���E	�.��z�A��*

A2S/average_reward_1���C��!,       ���E	^���z�A�*

A2S/average_reward_1���C��r�,       ���E	<8"��z�A��*

A2S/average_reward_1���CZ�k�,       ���E	*�&��z�A��*

A2S/average_reward_1�L�C
�6&,       ���E	��2��z�A��*

A2S/average_reward_1H!�C�J��,       ���E	t9��z�A��*

A2S/average_reward_1���C�<�,       ���E	��>��z�A��*

A2S/average_reward_1��C]�t,       ���E	��D��z�Aɜ*

A2S/average_reward_13��CB6e@,       ���E	U�K��z�Aל*

A2S/average_reward_1 �C�J��,       ���E	�T��z�A�*

A2S/average_reward_1
ױC�qk6,       ���E	�~5��z�AΤ*

A2S/average_reward_1�޴CH���,       ���E	��:��z�Aפ*

A2S/average_reward_1�вC�#N,       ���E	S!��z�A��*

A2S/average_reward_1�յC\��,       ���E	b��z�Aʬ*

A2S/average_reward_1�Q�C��
,       ���E	Y�#��z�Aܬ*

A2S/average_reward_1)|�C�g�,       ���E	��/��z�A��*

A2S/average_reward_1
w�C���W,       ���E	�4;��z�A��*

A2S/average_reward_1ff�CIY�K,       ���E	��A��z�A��*

A2S/average_reward_1  �CɘSN,       ���E	��I��z�A��*

A2S/average_reward_1
w�C`J,       ���E	nS��z�Aí*

A2S/average_reward_1{�Ck澡,       ���E	E�Q��z�A��*

A2S/average_reward_1쑨C7K!,       ���E	/W��z�A��*

A2S/average_reward_13��C�V,       ���E	�=^��z�Aŵ*

A2S/average_reward_1
��C�/(�,       ���E	�Cc��z�Aϵ*

A2S/average_reward_1�H�CI׺ ,       ���E	��i��z�Aܵ*

A2S/average_reward_1fF�C�ͫ,       ���E	�Hp��z�A�*

A2S/average_reward_1=�Ch��D,       ���E	��u��z�A��*

A2S/average_reward_1)��C��J,       ���E	f{��z�A��*

A2S/average_reward_1���CQ���,       ���E	����z�A��*

A2S/average_reward_13S�C�v��,       ���E	�M���z�A��*

A2S/average_reward_1 @�C&�a9,       ���E	�苯�z�A��*

A2S/average_reward_1��C�>,       ���E	�;���z�A��*

A2S/average_reward_1���C�Q��,       ���E	*����z�AѶ*

A2S/average_reward_1f�CB��,       ���E	4x���z�A޶*

A2S/average_reward_1ͬ�Co���,       ���E	z����z�A��*

A2S/average_reward_1\ψC��1I,       ���E	U���z�A��*

A2S/average_reward_1q��CF��,       ���E	�伯�z�A��*

A2S/average_reward_1���Cx`=��       �v@�	+ ��z�A��*�

A2S/klFѩ>

A2S/average_advantageO?/?

A2S/policy_network_losst[�>

A2S/value_network_loss�<�C

A2S/q_network_loss�ʭC��\,       ���E	����z�A��*

A2S/average_reward_1�p�C*�$,       ���E	^u��z�A��*

A2S/average_reward_1q=�C5ĭ,       ���E		���z�A��*

A2S/average_reward_1f�C���k,       ���E	V���z�A��*

A2S/average_reward_1�ˈCup�,       ���E	����z�A��*

A2S/average_reward_1�ڃC��<,       ���E	Iy��z�A��*

A2S/average_reward_1ᚈC�b-,       ���E	U���z�A��*

A2S/average_reward_1H��C)&O�,       ���E	LS���z�A��*

A2S/average_reward_1�>�C<b@�,       ���E	�Y���z�A��*

A2S/average_reward_1�>�CҼ ,       ���E	p� ��z�A��*

A2S/average_reward_1N�Cq��3,       ���E	"n��z�A��*

A2S/average_reward_1
�CV�#,       ���E	�����z�A��*

A2S/average_reward_1
�C����,       ���E	�e��z�A��*

A2S/average_reward_13ӂC����,       ���E	Ŵ��z�A��*

A2S/average_reward_1f��C1[�,       ���E	����z�A��*

A2S/average_reward_1\�|C�0,       ���E	����z�A��*

A2S/average_reward_1�rC�',       ���E	6��z�A��*

A2S/average_reward_1ErC�^�o,       ���E	����z�A��*

A2S/average_reward_1HahC�L��,       ���E	����z�A��*

A2S/average_reward_1q}`C����,       ���E	�J#��z�A��*

A2S/average_reward_1��_C�Z��,       ���E	�(��z�A��*

A2S/average_reward_1�_CX�!�,       ���E	\�,��z�A��*

A2S/average_reward_1��UC7ʴ,       ���E	E���z�A��*

A2S/average_reward_1�:_C<�1n,       ���E	[*��z�A��*

A2S/average_reward_1�QUCݭ�#,       ���E	����z�A��*

A2S/average_reward_13�^C�h�,       ���E	����z�A��*

A2S/average_reward_1f�^C�"��,       ���E	�~��z�A��*

A2S/average_reward_133hC��C,       ���E	�����z�A��*

A2S/average_reward_1  hCt��,       ���E	�H���z�A��*

A2S/average_reward_1�^C�ҿW,       ���E	;q���z�A��*

A2S/average_reward_1f�]C�m�,       ���E	q���z�A��*

A2S/average_reward_1�TC��2�,       ���E		���z�A��*

A2S/average_reward_1)JC�|*,       ���E	����z�A��*

A2S/average_reward_1��ICo�V,       ���E	����z�A��*

A2S/average_reward_1��ICY�j ,       ���E	����z�A��*

A2S/average_reward_1�zIC��ȝ,       ���E	u[���z�A��*

A2S/average_reward_1�ICFӏN,       ���E	����z�A��*

A2S/average_reward_1.?C���q,       ���E	r6
��z�A��*

A2S/average_reward_1��8C龝,       ���E	����z�A��*

A2S/average_reward_1��.C���,       ���E	kJ��z�A��*

A2S/average_reward_1��$C�I,       ���E	;���z�A��*

A2S/average_reward_1�Q$C+O�g,       ���E	��!��z�A��*

A2S/average_reward_1�kC�{��,       ���E	��;��z�A��*

A2S/average_reward_1
�C��+,       ���E	�TC��z�A��*

A2S/average_reward_1�QC����,       ���E	�����z�A��*

A2S/average_reward_1)�CU�^,       ���E	�r��z�A��*

A2S/average_reward_1Rx$C(	�w,       ���E	�J��z�A߉*

A2S/average_reward_1�.CϜ��,       ���E	�$O��z�A�*

A2S/average_reward_1
�-C4��,       ���E	oAT��z�A��*

A2S/average_reward_13�#C�Y,       ���E	��X��z�A��*

A2S/average_reward_1�#Cc�n,       ���E	�;_��z�A��*

A2S/average_reward_1=
#C--K�,       ���E	�g��z�A��*

A2S/average_reward_1\#Ck�wi,       ���E	ϧm��z�A��*

A2S/average_reward_1\#C�P&,       ���E	�
s��z�A��*

A2S/average_reward_1#C�CP�,       ���E	=Dx��z�A��*

A2S/average_reward_1H!C��h�,       ���E	�f��z�A��*

A2S/average_reward_1��"C���|,       ���E	o,l��z�A��*

A2S/average_reward_1��"CLr,       ���E	Lhq��z�A��*

A2S/average_reward_1��"C�N�?,       ���E	�_��z�A��*

A2S/average_reward_1q�,C���,       ���E	X
d��z�A��*

A2S/average_reward_1õ,C�v`#�       �v@�	M���z�A��*�

A2S/kl֧>

A2S/average_advantageiv1>

A2S/policy_network_loss�\ͽ

A2S/value_network_loss�s�C

A2S/q_network_loss�~�C0|��,       ���E	����z�A��*

A2S/average_reward_1��,C��&:,       ���E	pݰ��z�A��*

A2S/average_reward_1�h6C�6�,       ���E	�:���z�A��*

A2S/average_reward_13s6C�Qq�,       ���E	I����z�Aˢ*

A2S/average_reward_1n6C�N�u,       ���E	����z�AӢ*

A2S/average_reward_1��,C���,       ���E	����z�A�*

A2S/average_reward_1Rx,C�?��,       ���E	�����z�A�*

A2S/average_reward_1�u,Ca��k,       ���E	}o���z�AӪ*

A2S/average_reward_1336C�2��,       ���E	�����z�Aߪ*

A2S/average_reward_1336C����,       ���E	�:���z�Aǲ*

A2S/average_reward_1
@CE�,       ���E	����z�Aϲ*

A2S/average_reward_1\@C��Ä,       ���E	�c���z�A۲*

A2S/average_reward_1=
@C���,       ���E	�y���z�A�*

A2S/average_reward_1@C6x�E,       ���E	����z�A��*

A2S/average_reward_1�B6C�XU,,       ���E	�|���z�A��*

A2S/average_reward_1��6C�H�,       ���E	?���z�A��*

A2S/average_reward_1�,CǮ��,       ���E	�:��z�A׳*

A2S/average_reward_1�,C�7��,       ���E	����z�A��*

A2S/average_reward_1 �6Cj�j�,       ���E	�����z�A˻*

A2S/average_reward_1)�6C�I��,       ���E	ͩ���z�A��*

A2S/average_reward_1)\@Ca7�,       ���E	�*���z�A��*

A2S/average_reward_1R8JC�ß�,       ���E	'g���z�A��*

A2S/average_reward_1�hJCȪ�H,       ���E	� ���z�A��*

A2S/average_reward_1�5TC#�8,       ���E	����z�A��*

A2S/average_reward_1RxJC�t�Q,       ���E	�t���z�A��*

A2S/average_reward_1RxJC ��[,       ���E	�u���z�A��*

A2S/average_reward_1�QTCӳ�,       ���E	����z�A��*

A2S/average_reward_1�QTC=��M,       ���E	���z�A��*

A2S/average_reward_1��TC��Z,       ���E	|����z�A��*

A2S/average_reward_1)�TCr�,       ���E	����z�A��*

A2S/average_reward_1{�TC���$,       ���E	�!���z�A��*

A2S/average_reward_1=�TCA���,       ���E	����z�A��*

A2S/average_reward_1{�TCQ1��,       ���E	�]���z�A��*

A2S/average_reward_1��TC��|[,       ���E	(����z�A��*

A2S/average_reward_1��TCP[��,       ���E	�����z�A��*

A2S/average_reward_1�hTCsy��,       ���E	9K���z�A��*

A2S/average_reward_1�^TCt���,       ���E	�����z�A��*

A2S/average_reward_1ffTC����,       ���E	g����z�A��*

A2S/average_reward_1�+^C�r,       ���E	�����z�A��*

A2S/average_reward_1f&^C��V,       ���E	L����z�A��*

A2S/average_reward_1��gC.�8j,       ���E	����z�A��*

A2S/average_reward_1��gC�{�X,       ���E	+����z�A��*

A2S/average_reward_1��gCl� ),       ���E	2����z�A��*

A2S/average_reward_1f�gC[i�6,       ���E	AE���z�A��*

A2S/average_reward_1q�]C�H�,       ���E	�y���z�A��*

A2S/average_reward_1
^C��`,       ���E	�����z�A��*

A2S/average_reward_1�+TCr)Y,       ���E	h���z�A��*

A2S/average_reward_1\^C���,       ���E	 ����z�A��*

A2S/average_reward_1�^C�߬,       ���E	 ���z�A��*

A2S/average_reward_1H!TCGag,       ���E	Mb���z�A��*

A2S/average_reward_1)TC��4�,       ���E	狹��z�A��*

A2S/average_reward_1q�]C�iNB,       ���E	{����z�A��*

A2S/average_reward_1�TC�h�,       ���E	<����z�Aք*

A2S/average_reward_1  ^C��sW,       ���E	ԍ���z�A��*

A2S/average_reward_1��]C�υ ,       ���E	�����z�A�*

A2S/average_reward_1�^C(�g�,       ���E	�.���z�A��*

A2S/average_reward_1=
^CT���,       ���E	����z�A��*

A2S/average_reward_1�^C�r,       ���E	����z�A��*

A2S/average_reward_1�^C�!��,       ���E	����z�A��*

A2S/average_reward_1�hCj��,       ���E	%���z�A��*

A2S/average_reward_1�hC��a�,       ���E	<��z�A�*

A2S/average_reward_1��qC�p,       ���E	�1���z�A��*

A2S/average_reward_1��qC�[U,       ���E	�����z�A��*

A2S/average_reward_1�hC��F,       ���E	#����z�A��*

A2S/average_reward_1H!hCǦ��,       ���E	j����z�A��*

A2S/average_reward_1�:^Cu9�I,       ���E	�?���z�A��*

A2S/average_reward_1�0^C��f,       ���E	�8���z�A��*

A2S/average_reward_1�0^C���a,       ���E	�{���z�A��*

A2S/average_reward_1.^C#N�,       ���E	Z
���z�A��*

A2S/average_reward_1�+^C1{,       ���E	�$���z�A��*

A2S/average_reward_1�#^CT5o�       �v@�	ű���z�A��*�

A2S/kl��{>

A2S/average_advantageکL>

A2S/policy_network_loss<\<

A2S/value_network_lossK	�C

A2S/q_network_lossH��C^��,       ���E	+����z�A��*

A2S/average_reward_1�^CCd��,       ���E	3����z�A͝*

A2S/average_reward_1�+^C�_*�,       ���E	T��z�A�*

A2S/average_reward_1�B^C��,       ���E	���z�A��*

A2S/average_reward_1�uTC� ��,       ���E	B�Z��z�A��*

A2S/average_reward_1=�UC� ��,       ���E	��^��z�A��*

A2S/average_reward_1��UC̀�,       ���E	��e��z�A��*

A2S/average_reward_1��UCyDw:,       ���E	�k��z�A��*

A2S/average_reward_1{�UC�9nW,       ���E	4ֿ��z�AĠ*

A2S/average_reward_1�BWC���,       ���E	X���z�A��*

A2S/average_reward_1��XC�KA,       ���E	4�l��z�A��*

A2S/average_reward_1HaZC�|>_,       ���E	��q��z�A��*

A2S/average_reward_1�^ZC�)�,       ���E	Z3���z�Aͤ*

A2S/average_reward_1��[C�s;�,       ���E	
����z�Aܤ*

A2S/average_reward_1��[CH��,       ���E	��-��z�A��*

A2S/average_reward_1R�[C�b�-,       ���E	��w��z�A��*

A2S/average_reward_13sSC�=��,       ���E	ev���z�A��*

A2S/average_reward_1=
KC��+c,       ���E	�����z�A��*

A2S/average_reward_1=
KC�U�,       ���E	l!��z�A�*

A2S/average_reward_1)�LC��/�,       ���E	p�(��z�A�*

A2S/average_reward_1��LC ��),       ���E	�%.��z�A��*

A2S/average_reward_1H�LC}��6,       ���E	4�2��z�A��*

A2S/average_reward_1͌LCX�\,       ���E	Ua8��z�A��*

A2S/average_reward_1=�LCD�T,       ���E	�<��z�A��*

A2S/average_reward_1��LC'L�W,       ���E	�����z�A��*

A2S/average_reward_1�NC�-r�,       ���E	�����z�AѬ*

A2S/average_reward_1��EC7R�,       ���E	����z�Aڬ*

A2S/average_reward_1��EC��,       ���E	R^���z�A�*

A2S/average_reward_1q}EC�Վ5,       ���E	r!��z�A��*

A2S/average_reward_1q�<C���d,       ���E	c�~��z�Aů*

A2S/average_reward_1��>C2���,       ���E	:����z�A֯*

A2S/average_reward_1q�>C�b$,       ���E	9����z�A�*

A2S/average_reward_1�c6C
�jU,       ���E	���z�A��*

A2S/average_reward_1��7C��r/,       ���E	v�c��z�A��*

A2S/average_reward_13�8C�'�,       ���E	�N���z�A�*

A2S/average_reward_1��9Cw���,       ���E	�R���z�A��*

A2S/average_reward_1\�9Cc~[�,       ���E	j����z�A��*

A2S/average_reward_1R8;C��#,       ���E	{q%��z�A��*

A2S/average_reward_1R�2C.x,       ���E	����z�A��*

A2S/average_reward_1��4C��(�,       ���E	-(���z�A��*

A2S/average_reward_1��*C���4,       ���E	�����z�A��*

A2S/average_reward_1��*C���\,       ���E	?ȭ��z�A��*

A2S/average_reward_1f�*C�Q��,       ���E	�Ƿ��z�A��*

A2S/average_reward_1R�*Cp��,       ���E	K%��z�A�*

A2S/average_reward_1=�,C���,       ���E	�zj��z�A��*

A2S/average_reward_1�.C	JA�,       ���E	�#���z�A�*

A2S/average_reward_1��/C��C�,       ���E	W-���z�A��*

A2S/average_reward_1��/C�̵j,       ���E	e���z�A��*

A2S/average_reward_1)'C� ��,       ���E	�9��z�A��*

A2S/average_reward_1
'CXgZW,       ���E	�	��z�A��*

A2S/average_reward_1.C��,       ���E	��#��z�A��*

A2S/average_reward_1{TCx�,       ���E	�"-��z�A��*

A2S/average_reward_1f&C��Cd,       ���E	�2��z�Aľ*

A2S/average_reward_1q=	C*��,       ���E	sS7��z�A̾*

A2S/average_reward_1\	CE�r,       ���E	��?��z�Aܾ*

A2S/average_reward_1)	C��V�,       ���E	�S���z�A�*

A2S/average_reward_1)� C=���,       ���E	����z�A��*

A2S/average_reward_1��Cq�m�,       ���E	W�B��z�A��*

A2S/average_reward_13�CYH�,       ���E	���z�A��*

A2S/average_reward_1�GCLn6s,       ���E	�[���z�A��*

A2S/average_reward_1�CNߧ�,       ���E	����z�A��*

A2S/average_reward_1��C��c�,       ���E	����z�A��*

A2S/average_reward_1{�C��X�,       ���E	��?��z�A��*

A2S/average_reward_1�^CW]�,       ���E	�\D��z�A��*

A2S/average_reward_1�QC4�,       ���E	�{G��z�A��*

A2S/average_reward_1=JCv8N,       ���E	�J��z�A��*

A2S/average_reward_1�5C�`�b,       ���E	uEQ��z�A��*

A2S/average_reward_133C[W$�,       ���E	o�T��z�A��*

A2S/average_reward_1\��B�:	�,       ���E	�0Z��z�A��*

A2S/average_reward_1���Bݓ.,       ���E	�M���z�A��*

A2S/average_reward_1��B�����       �v@�	�����z�A��*�

A2S/kl�(B>

A2S/average_advantage��<

A2S/policy_network_lossٮ=

A2S/value_network_loss��	B

A2S/q_network_loss.�	B��m,       ���E	�8���z�A��*

A2S/average_reward_1�B�B��_,       ���E	!����z�A��*

A2S/average_reward_1=�	C��0�,       ���E	����z�A��*

A2S/average_reward_1nCs!�,       ���E	����z�A��*

A2S/average_reward_1
WC�6$�,       ���E	����z�A��*

A2S/average_reward_1�'CA�
,       ���E	�a���z�A��*

A2S/average_reward_1=
1C��P�,       ���E	=c��z�A��*

A2S/average_reward_1=
1C��l�,       ���E	��z�A�*

A2S/average_reward_13�:C�J/ ,       ���E	?��z�A֎*

A2S/average_reward_1��DC���T,       ���E	����z�A��*

A2S/average_reward_1�NCm�l,       ���E	�b�	�z�A��*

A2S/average_reward_1�NC��4,       ���E	�9��z�A��*

A2S/average_reward_1ףXCy@34,       ���E	����z�A��*

A2S/average_reward_1ףXCsy�,       ���E	a���z�A޵*

A2S/average_reward_1=�bC[�l>,       ���E	���z�Aƽ*

A2S/average_reward_1�YlC�,       ���E	�Tj�z�A��*

A2S/average_reward_1�BvC =�N,       ���E	�m�z�A��*

A2S/average_reward_1��CMh�f,       ���E	�S�z�A��*

A2S/average_reward_1���C��~*,       ���E	�2r�z�A��*

A2S/average_reward_1���C�@�',       ���E	f�G�z�A��*

A2S/average_reward_1\�C�1ф,       ���E	��C�z�A��*

A2S/average_reward_1\�C�U�,       ���E	m�$�z�A��*

A2S/average_reward_1�ގC�g�V,       ���E	��#!�z�A��*

A2S/average_reward_13ӓC�w��,       ���E	ӄ8#�z�A�*

A2S/average_reward_1ẘCF�L,       ���E	�%%�z�A֋*

A2S/average_reward_1��C�.��,       ���E	�7'�z�A��*

A2S/average_reward_1ף�C�K3,       ���E	�(�z�A��*

A2S/average_reward_1ף�C7V��,       ���E	w+�z�A��*

A2S/average_reward_1Õ�C����,       ���E	�#�,�z�A��*

A2S/average_reward_1���C�-�,       ���E	�S�.�z�A޲*

A2S/average_reward_1)|�C@>3,       ���E	�!�0�z�Aƺ*

A2S/average_reward_1�p�C�]9,       ���E	�3�2�z�A��*

A2S/average_reward_1q]�C�~4,       ���E	fa�4�z�A��*

A2S/average_reward_1�C�C���\,       ���E	Q��6�z�A��*

A2S/average_reward_1=*�C�ako,       ���E	�?�8�z�A��*

A2S/average_reward_1�q�CMaY�,       ���E	v�:�z�A��*

A2S/average_reward_1ff�C��#,       ���E	�2�<�z�A��*

A2S/average_reward_13S�C�_l ,       ���E	�!�>�z�A��*

A2S/average_reward_1E�C=���,       ���E	5��@�z�A��*

A2S/average_reward_1q}�C��F�,       ���E	vq�B�z�A�*

A2S/average_reward_1
��C�.i�,       ���E	�h�D�z�Aֈ*

A2S/average_reward_1\��C1�S�,       ���E	�M�F�z�A��*

A2S/average_reward_1���C��,       ���E	&b�H�z�A��*

A2S/average_reward_1���C1	E/,       ���E	g�J�z�A��*

A2S/average_reward_1\��C1~
,       ���E	:L�L�z�A��*

A2S/average_reward_1��Cx�I",       ���E	��N�z�Aޯ*

A2S/average_reward_1E�C�~,,       ���E	��P�z�AƷ*

A2S/average_reward_1�y�C��\�,       ���E	��wR�z�A��*

A2S/average_reward_1�5D�(�`,       ���E	��`T�z�A��*

A2S/average_reward_1=JD�fI,       ���E	�aV�z�A��*

A2S/average_reward_1��D�U�',       ���E	��dX�z�A��*

A2S/average_reward_1=:	DJ#�A,       ���E	~IZ�z�A��*

A2S/average_reward_1{�DJ��%,       ���E	ĞT\�z�A��*

A2S/average_reward_1�,D�!
�,       ���E	�%@^�z�A��*

A2S/average_reward_1��DV2�,       ���E	��T`�z�A��*

A2S/average_reward_1\�DY+�,       ���E	�FXb�z�A��*

A2S/average_reward_1\�D8Cb�,       ���E	I�Dd�z�Aօ	*

A2S/average_reward_1�YDM�:),       ���E	�,=f�z�A��	*

A2S/average_reward_13�D%��,       ���E	-h�z�A��	*

A2S/average_reward_13�D���w,       ���E	�j�z�A��	*

A2S/average_reward_1H�Dt���,       ���E	!&�k�z�A��	*

A2S/average_reward_1ff D��E,       ���E	���m�z�Aެ	*

A2S/average_reward_1͌"D�g�J,       ���E	H��o�z�Aƴ	*

A2S/average_reward_1�$DSuH,       ���E	���q�z�A��	*

A2S/average_reward_1\�&Da3��,       ���E	�s�z�A��	*

A2S/average_reward_1�)D�O�,       ���E	(z�u�z�A��	*

A2S/average_reward_1�+D���,       ���E	���w�z�A��	*

A2S/average_reward_1��-D�Ø,       ���E	��y�z�A��	*

A2S/average_reward_1��/D%�a,,       ���E	Ԅ�{�z�A��	*

A2S/average_reward_1\�1D��x,       ���E	���}�z�A��	*

A2S/average_reward_1�94D3e�,       ���E	·��z�A��	*

A2S/average_reward_1��6D����,       ���E	zx��z�A��	*

A2S/average_reward_1�)9Df��a,       ���E	�xm��z�Aւ
*

A2S/average_reward_1͜;D��Pc,       ���E	��^��z�A��
*

A2S/average_reward_1��=D�cצ,       ���E	�f��z�A��
*

A2S/average_reward_1
�?D	�:,       ���E	��L��z�A��
*

A2S/average_reward_13�AD\�,       ���E	?�L��z�A��
*

A2S/average_reward_1�)DD׳�H,       ���E	�v[��z�Aީ
*

A2S/average_reward_1�LFD"���,       ���E	Ì���z�AƱ
*

A2S/average_reward_1f�HDh�[�,       ���E	1hk��z�A��
*

A2S/average_reward_1�@KD����       �v@�	K�P��z�A��
*�

A2S/kld+>

A2S/average_advantage=

A2S/policy_network_lossc8�;

A2S/value_network_loss.8�C

A2S/q_network_loss��CP���,       ���E	�N���z�A��
*

A2S/average_reward_1��KDl�,       ���E	a�2��z�A��
*

A2S/average_reward_1�XLDf�'�,       ���E	=`���z�A��
*

A2S/average_reward_1��LD����,       ���E	�B#��z�A��
*

A2S/average_reward_1ÕMDG�ʌ,       ���E	�-���z�A��
*

A2S/average_reward_1R8NDd�J,       ���E	�e��z�A��
*

A2S/average_reward_1\oND����,       ���E	~ꗕ�z�A��
*

A2S/average_reward_1��NDZY�,       ���E	��$��z�A��
*

A2S/average_reward_1=�NDQ�Y ,       ���E	��z�A��
*

A2S/average_reward_1fODel,       ���E	l�:��z�A��
*

A2S/average_reward_1�UODRw�r,       ���E	O�Ɨ�z�A��
*

A2S/average_reward_13�OD�栦,       ���E	��M��z�A��
*

A2S/average_reward_13�PD��N,       ���E	ٝΘ�z�A��
*

A2S/average_reward_1H�PDlAK,       ���E	T�R��z�A��
*

A2S/average_reward_1
WQDؿ/�,       ���E	��љ�z�A��
*

A2S/average_reward_1R�QDi*],       ���E	8/C��z�A��
*

A2S/average_reward_13�RD����,       ���E	i����z�A��
*

A2S/average_reward_1�SD���,       ���E	,5��z�A��
*

A2S/average_reward_1͜SD�WP3,       ���E	�x���z�A��
*

A2S/average_reward_1\?TDӇIh,       ���E	��A��z�A��
*

A2S/average_reward_1H�TDj�bg,       ���E	ผ�z�A��
*

A2S/average_reward_1��RDg�E�,       ���E	���z�A��
*

A2S/average_reward_1)�PD&Ta�,       ���E	o���z�A��
*

A2S/average_reward_13�ND�o��,       ���E	=���z�A��
*

A2S/average_reward_1��LDg��,       ���E	]���z�A��
*

A2S/average_reward_1RKD���;,       ���E	���z�A��
*

A2S/average_reward_1�@ID��,       ���E	O����z�A��
*

A2S/average_reward_1�kGDa(߱,       ���E	����z�A��
*

A2S/average_reward_1H�ED��,       ���E	d��z�A��
*

A2S/average_reward_1��CD�kJ,       ���E	�V��z�A��
*

A2S/average_reward_1׳AD�kؽ,       ���E	+/���z�A��
*

A2S/average_reward_1 �?Db�,       ���E	����z�A��
*

A2S/average_reward_1�=D���6,       ���E	�����z�A��
*

A2S/average_reward_1�<D4�,       ���E	�U��z�A��
*

A2S/average_reward_13C:D��w�,       ���E	�w���z�A��
*

A2S/average_reward_1�L8D$K :,       ���E	9��z�A��
*

A2S/average_reward_1�[6D9�|W,       ���E	-����z�A��*

A2S/average_reward_13�4D�L��,       ���E	�S���z�A��*

A2S/average_reward_1͜2Dw�,       ���E	!y��z�A�*

A2S/average_reward_1�0D��{,       ���E	ܜ���z�A��*

A2S/average_reward_1�.DumDN,       ���E	4o��z�AՈ*

A2S/average_reward_1��,D&=�,       ���E	�Q��z�A��*

A2S/average_reward_1
�*D�9�,       ���E	tX��z�A��*

A2S/average_reward_1
)D��,       ���E	z�ѧ�z�A��*

A2S/average_reward_1=*'DMM,       ���E	]�D��z�A��*

A2S/average_reward_1�A%D	5t,       ���E	�A���z�A�*

A2S/average_reward_1\O#D���,       ���E	��/��z�Aۓ*

A2S/average_reward_1�i!DI�&n,       ���E	㶭��z�Aߕ*

A2S/average_reward_1 �D��)�,       ���E	-W+��z�Aʗ*

A2S/average_reward_1f�Dd{w�,       ���E	"����z�A��*

A2S/average_reward_1\�D��,       ���E	�<��z�A��*

A2S/average_reward_1
�D��m],       ���E	�栫�z�A��*

A2S/average_reward_1��D^9}�,       ���E	P���z�A��*

A2S/average_reward_1=
Da���,       ���E	O����z�A��*

A2S/average_reward_1
'D�Br,       ���E	C��z�A�*

A2S/average_reward_1�;D��u ,       ���E	mfz��z�AΤ*

A2S/average_reward_1�QDx��,       ���E	�����z�AϦ*

A2S/average_reward_1fvDo~�,       ���E	P�v��z�A��*

A2S/average_reward_1͌D��}L,       ���E	�q���z�Aê*

A2S/average_reward_1f�
DA��,       ���E	�@���z�A��*

A2S/average_reward_1R�D���,       ���E	�����z�A��*

A2S/average_reward_1��D�V|�,       ���E	�1a��z�A��*

A2S/average_reward_1��D�I+�,       ���E	I���z�A��*

A2S/average_reward_1{D��5�,       ���E	Qi`��z�A�*

A2S/average_reward_1�1D6=�,       ���E	`#ϱ�z�Aе*

A2S/average_reward_1���C���,       ���E	��R��z�A˷*

A2S/average_reward_1���C���,       ���E	�eȲ�z�A��*

A2S/average_reward_1��Cb3k,       ���E	�B��z�AĻ*

A2S/average_reward_1\O�C.1
,       ���E	|����z�A��*

A2S/average_reward_13��C��1	,       ���E	�76��z�A��*

A2S/average_reward_1��C�q��,       ���E	à���z�A��*

A2S/average_reward_1���C�rW�,       ���E	��+��z�A��*

A2S/average_reward_1�5�C���v,       ���E	�⳵�z�A��*

A2S/average_reward_1���C)R�,       ���E	�G��z�A��*

A2S/average_reward_1���C��@,       ���E	/���z�A��*

A2S/average_reward_1f�C�h��,       ���E	�R5��z�A��*

A2S/average_reward_1�Q�C7>j,       ���E	����z�A��*

A2S/average_reward_1H��C)��,       ���E	��(��z�A��*

A2S/average_reward_1 ��C%�	�,       ���E	hݴ��z�A��*

A2S/average_reward_1��CNL;~,       ���E	R/<��z�A��*

A2S/average_reward_1\O�C���       �v@�	��{��z�A��*�

A2S/klf�>

A2S/average_advantage��?

A2S/policy_network_loss>�8�

A2S/value_network_loss���A

A2S/q_network_loss�DA"�ƛ,       ���E	��ʹ�z�A��*

A2S/average_reward_1H!�Cq/�,       ���E	�`��z�A��*

A2S/average_reward_1��C�"ӕ,       ���E	�m��z�A��*

A2S/average_reward_1���Cc�,       ���E	���z�A��*

A2S/average_reward_1q}�Ci�a",       ���E	ܭ��z�A��*

A2S/average_reward_1�Q�C/��f,       ���E	/�P��z�A��*

A2S/average_reward_1���C&�N,       ���E	��z�A��*

A2S/average_reward_1 ��C�5�,       ���E	����z�A��*

A2S/average_reward_1�l�C���,       ���E	��K��z�A��*

A2S/average_reward_1E�C�~P�,       ���E	�����z�A��*

A2S/average_reward_1��C6@�@,       ���E	Nqݼ�z�A��*

A2S/average_reward_1 ��CM9�,       ���E	e1��z�A��*

A2S/average_reward_1 ��C�풖,       ���E	y���z�A��*

A2S/average_reward_1�G�C6֧R,       ���E	�u½�z�A��*

A2S/average_reward_1{�C��P,       ���E	����z�A��*

A2S/average_reward_1�C
`H,       ���E	2�m��z�A��*

A2S/average_reward_1{t�Cdhf�,       ���E	z����z�A��*

A2S/average_reward_1�B|C��,       ���E	�i��z�A��*

A2S/average_reward_1��sC�U�@,       ���E	��U��z�A��*

A2S/average_reward_1�^kC1���,       ���E	�J���z�A��*

A2S/average_reward_1q�bC��zF,       ���E	xH��z�A��*

A2S/average_reward_1�bC[K�6,       ���E	+��z�A��*

A2S/average_reward_1{TaC�&�,       ���E	=}��z�A��*

A2S/average_reward_1�u`CB�*�,       ���E	�5���z�A��*

A2S/average_reward_1 @_C�:F�,       ���E	���z�A��*

A2S/average_reward_1�5^CFk,       ���E	�Tr��z�A��*

A2S/average_reward_1\�]Cp$�,       ���E	]Ѿ��z�A��*

A2S/average_reward_1Rx\C��,       ���E	p	��z�A��*

A2S/average_reward_1�5[C�]-H,       ���E	�_��z�A��*

A2S/average_reward_1�5ZCݖRo,       ���E	NT���z�A��*

A2S/average_reward_133YCtJd�,       ���E	�c���z�A��*

A2S/average_reward_1f�WC͕'�,       ���E	��U��z�A��*

A2S/average_reward_1��VC��,       ���E	�n���z�A��*

A2S/average_reward_1��UCX�֎,       ���E	�8���z�A��*

A2S/average_reward_1�TCrIF,       ���E	<���z�A��*

A2S/average_reward_1��SC@��,       ���E	�j��z�A��*

A2S/average_reward_1õRCb���,       ���E	�s���z�A��*

A2S/average_reward_1�QC�5�,       ���E	����z�A��*

A2S/average_reward_1�(QC2D�,       ���E	NO��z�A�*

A2S/average_reward_1{�OC~�E,       ���E	者��z�A��*

A2S/average_reward_1\�NCӜ~�,       ���E	u����z�A��*

A2S/average_reward_1q}MC�J4�,       ���E	��"��z�A��*

A2S/average_reward_1q}LCZB�,       ���E	;V{��z�A��*

A2S/average_reward_1\�KC�',       ���E	�����z�A��*

A2S/average_reward_1H�JC���,       ���E	i��z�A̇*

A2S/average_reward_1��IC�BN�,       ���E	]AO��z�Aۈ*

A2S/average_reward_1ffHC���5,       ���E	v����z�A��*

A2S/average_reward_1�YGC.V:F,       ���E	����z�A��*

A2S/average_reward_1
�FC5���,       ���E	e'��z�A��*

A2S/average_reward_1�zEC�U��,       ���E	�W{��z�A��*

A2S/average_reward_1q}DC��Bv,       ���E	�d���z�A*

A2S/average_reward_1
�CC�Ę,       ���E	G/-��z�A�*

A2S/average_reward_1��BC���,       ���E	3�r��z�A�*

A2S/average_reward_1�uAC[R��,       ���E	N����z�A��*

A2S/average_reward_1 �@C�gP",       ���E	�'��z�A��*

A2S/average_reward_1��?C�8�,       ���E	T9^��z�AҔ*

A2S/average_reward_1 @?C��,       ���E	� ���z�A��*

A2S/average_reward_1�G>CƆ��,       ���E	�v���z�A��*

A2S/average_reward_1R8=C�EA,       ���E	Zq7��z�A��*

A2S/average_reward_1�u<Cu�S�,       ���E	�Ԃ��z�A��*

A2S/average_reward_1�;CQ?�,       ���E	b����z�A̚*

A2S/average_reward_1=�:CBd)�,       ���E	j���z�A�*

A2S/average_reward_1):C��c�,       ���E	�Di��z�A��*

A2S/average_reward_1�z9CV̮J,       ���E	@ ���z�A��*

A2S/average_reward_1\�8C�>S,       ���E	FY��z�A۟*

A2S/average_reward_1��7C���~,       ���E	�_W��z�A�*

A2S/average_reward_1H!7C"��,       ���E	�����z�A��*

A2S/average_reward_1�6C��,       ���E	|z���z�A��*

A2S/average_reward_1  5C�F�8,       ���E	�A��z�A��*

A2S/average_reward_1�B4C�*O�,       ���E	%4��z�A��*

A2S/average_reward_1�k3C�UN,       ���E	%����z�Aæ*

A2S/average_reward_1�52C\K��,       ���E	^s��z�A�*

A2S/average_reward_1�k1C�RI,       ���E	�b��z�A�*

A2S/average_reward_1\O0CVTg,       ���E	#���z�A�*

A2S/average_reward_1�B/C���,       ���E	J���z�A��*

A2S/average_reward_1�z.C���1,       ���E	��6��z�A��*

A2S/average_reward_1\�-C&�gy,       ���E	����z�A��*

A2S/average_reward_1\�,C#�V�,       ���E	�m���z�Aʮ*

A2S/average_reward_1�+C��>D,       ���E	����z�A�*

A2S/average_reward_1)�*CKA!u,       ���E	/�b��z�A��*

A2S/average_reward_1��)C�K��,       ���E	�f���z�A��*

A2S/average_reward_1\)C��8,       ���E	e���z�Aг*

A2S/average_reward_1��(C���P,       ���E	��`��z�A�*

A2S/average_reward_1�'Ch?!�,       ���E	����z�A��*

A2S/average_reward_1=�&CR|�,       ���E	i���z�A��*

A2S/average_reward_1H�%C↑�,       ���E	��J��z�A��*

A2S/average_reward_1��$C8m��,       ���E	횔��z�Aɹ*

A2S/average_reward_1�#C�x,       ���E	�K���z�A�*

A2S/average_reward_1��"C�b>,       ���E	�o:��z�A��*

A2S/average_reward_1f&"C��]�,       ���E	�ɏ��z�A��*

A2S/average_reward_13s!C,I���       �v@�	7����z�A��*�

A2S/kl�	>

A2S/average_advantage�&�

A2S/policy_network_loss�;

A2S/value_network_loss~��?

A2S/q_network_loss���?����,       ���E	�K���z�A��*

A2S/average_reward_1��(C�"��,       ���E	HÆ��z�A��*

A2S/average_reward_1q}0C�+�,       ���E	�f��z�A��*

A2S/average_reward_1��7C)s�,       ���E	�o��z�A��*

A2S/average_reward_1�#?Cȁ�q,       ���E	.o��z�A��*

A2S/average_reward_1)�FC��W�,       ���E	�b��z�A��*

A2S/average_reward_1ENC{Y�,       ���E	_+p��z�A��*

A2S/average_reward_1f�UCD���,       ���E	oS_��z�A��*

A2S/average_reward_1�h]C3��',       ���E	�(x��z�A�*

A2S/average_reward_1��dC�%�,       ���E	�Uz��z�Aɋ*

A2S/average_reward_1=JlCx+��,       ���E	T{n��z�A��*

A2S/average_reward_1f�tC��i~,       ���E	7_u��z�A��*

A2S/average_reward_1�}C3B9,       ���E	jZ��z�A��*

A2S/average_reward_1�ÂC��L4,       ���E	_�Z��z�A�*

A2S/average_reward_1
��C�LƲ,       ���E	D�I��z�AѲ*

A2S/average_reward_1�"�CU�C,       ���E	\�B��z�A��*

A2S/average_reward_1�~�C]��,       ���E	ֻ.��z�A��*

A2S/average_reward_1{��Cd�3�,       ���E	Yr!��z�A��*

A2S/average_reward_1��C���N,       ���E	 ^���z�A��*

A2S/average_reward_1\/�Cbkv",       ���E	F?���z�A��*

A2S/average_reward_1�g�C��,       ���E	;����z�A��*

A2S/average_reward_1{��C�B�~,       ���E	�;���z�A��*

A2S/average_reward_1{��C-f�V,       ���E	���z�A��*

A2S/average_reward_1�,�C����,       ���E	ʤ��z�A��*

A2S/average_reward_1 ��C�y��,       ���E	=���z�A�*

A2S/average_reward_1챵C����,       ���E	����z�AɈ*

A2S/average_reward_1  �C��<u,       ���E	��	�z�A��*

A2S/average_reward_13S�C|�ã,       ���E	��z�A��*

A2S/average_reward_1�~�C���,       ���E	���z�A��*

A2S/average_reward_1��C�;�,       ���E	J��z�A�*

A2S/average_reward_1��C�۸6,       ���E	zh��z�Aѯ*

A2S/average_reward_1 @�C'>��,       ���E	G�{�z�A��*

A2S/average_reward_1n�C�h�	,       ���E	a/m�z�A��*

A2S/average_reward_1{��CB��,       ���E	V(e�z�A��*

A2S/average_reward_1���C�|��,       ���E	<�d�z�A��*

A2S/average_reward_1{�C�Lh<,       ���E	�C�z�A��*

A2S/average_reward_1�9�C	Yz,,       ���E	��?�z�A��*

A2S/average_reward_1\o�C�� ,       ���E	A��z�A��*

A2S/average_reward_1H��C��V`,       ���E	�d!�z�A��*

A2S/average_reward_1\��C���r,       ���E	���"�z�A��*

A2S/average_reward_1�'�C9ң�,       ���E	�%�z�A��*

A2S/average_reward_1H��C��,       ���E	R�'�z�AɅ*

A2S/average_reward_1)��C��ٞ,       ���E	h�)�z�A��*

A2S/average_reward_1�D��o�,       ���E	�+�z�A��*

A2S/average_reward_1�0D���,       ���E	�-�z�A��*

A2S/average_reward_1�WD+�E,       ���E	CA/�z�A�*

A2S/average_reward_1)|D|��,       ���E	��1�z�AѬ*

A2S/average_reward_1ד	D��+,       ���E	���2�z�A��*

A2S/average_reward_1
�D����,       ���E	mS�4�z�A��*

A2S/average_reward_1H�D� -�,       ���E	g?�6�z�A��*

A2S/average_reward_1D���,       ���E	���8�z�A��*

A2S/average_reward_1),D1I��,       ���E	�X�:�z�A��*

A2S/average_reward_1�^D�g[,       ���E	���<�z�A��*

A2S/average_reward_13sD4�6�,       ���E	m��>�z�A��*

A2S/average_reward_1
�D���',       ���E	K��@�z�A��*

A2S/average_reward_1��D����,       ���E	���B�z�A��*

A2S/average_reward_1 �D���$,       ���E	"E�D�z�A��*

A2S/average_reward_1R�D��x,       ���E	�w�F�z�Aɂ*

A2S/average_reward_13!D3н�,       ���E	1��H�z�A��*

A2S/average_reward_1�B#D��A,       ���E	g��J�z�A��*

A2S/average_reward_1
g%DD1�#,       ���E	���L�z�A��*

A2S/average_reward_1{�'D�`��,       ���E	_��N�z�A�*

A2S/average_reward_1��)DpP�,       ���E	� �P�z�Aѩ*

A2S/average_reward_1��+D%���,       ���E	�>�R�z�A��*

A2S/average_reward_1
�-D<K�T,       ���E	%�T�z�A��*

A2S/average_reward_1��/DwI��,       ���E	nv�V�z�A��*

A2S/average_reward_1�2D���,       ���E	6ѶX�z�A��*

A2S/average_reward_1%4D/�,       ���E	��Z�z�A��*

A2S/average_reward_1\O6D�)�,       ���E	�m�\�z�A��*

A2S/average_reward_1�q8D�~�,       ���E	v��^�z�A��*

A2S/average_reward_1�:DsX�,       ���E	X��`�z�A��*

A2S/average_reward_1
�<D� 4n,       ���E	3��b�z�A��*

A2S/average_reward_1)�>D+V��,       ���E	?�d�z�A��*

A2S/average_reward_1{�@D�ѯ�,       ���E	��f�z�A��*

A2S/average_reward_1)�BDSc��,       ���E	!T�h�z�A��*

A2S/average_reward_1�ED�G_,       ���E	�i�j�z�A��*

A2S/average_reward_1�"GD����,       ���E	�ʞl�z�A��*

A2S/average_reward_1)LID�ӿ,       ���E	!5�n�z�A�*

A2S/average_reward_1=jKD�\�,       ���E	X��p�z�AѦ*

A2S/average_reward_13�MD<�֓,       ���E	~�r�z�A��*

A2S/average_reward_1 �OD���,       ���E	o�t�z�A��*

A2S/average_reward_1��QDWC,       ���E	a�yv�z�A��*

A2S/average_reward_1��SD�'�,       ���E	���x�z�A��*

A2S/average_reward_1)VD��,       ���E	2�}z�z�A��*

A2S/average_reward_1�BXD��^,       ���E	�Qg|�z�A��*

A2S/average_reward_1 `ZD(��|,       ���E	H�_~�z�A��*

A2S/average_reward_1{�\D��u�,       ���E	�>e��z�A��*

A2S/average_reward_1 �^D��,�,       ���E	>SN��z�A��*

A2S/average_reward_1��`Dާ�,       ���E	u�:��z�A��*

A2S/average_reward_1��bD?�+�,       ���E	:�"��z�A��*

A2S/average_reward_1��dD��a�,       ���E	ұ)��z�A��*

A2S/average_reward_1qgDկ�b,       ���E	^���z�A��*

A2S/average_reward_1{$iD�r��,       ���E	0[���z�A��*

A2S/average_reward_1HAkDd�=,       ���E	�� ��z�A�*

A2S/average_reward_1�emD��`&,       ���E	����z�Aѣ*

A2S/average_reward_1�oD����,       ���E	����z�A��*

A2S/average_reward_1ףqD�M),       ���E	����z�A��*

A2S/average_reward_1q�sD�\P
,       ���E	��*��z�A��*

A2S/average_reward_1��uD�-�_,       ���E	C���z�A��*

A2S/average_reward_1
�wD~�C,       ���E	����z�A��*

A2S/average_reward_1  zD��ϗ       �v@�	Z�;��z�A��*�

A2S/kl&�=

A2S/average_advantage�-�

A2S/policy_network_loss��.�

A2S/value_network_loss���C

A2S/q_network_loss��C��i,       ���E	�%��z�A��*

A2S/average_reward_1��wD|.��,       ���E	b���z�A��*

A2S/average_reward_1��uD\�.,       ���E	�9��z�A��*

A2S/average_reward_1��sDQ�C�,       ���E	�ZL��z�A��*

A2S/average_reward_1=ZqD�,       ���E	)Ԗ��z�A��*

A2S/average_reward_1�2oD���,       ���E	�pٜ�z�A��*

A2S/average_reward_1�	mD�z�,       ���E	����z�A��*

A2S/average_reward_1\�jD�`,       ���E	�`i��z�A��*

A2S/average_reward_1õhD���,       ���E	����z�A��*

A2S/average_reward_1 �fD�}�,       ���E	�K��z�A��*

A2S/average_reward_1�edD.C�L,       ���E	��]��z�A��*

A2S/average_reward_1\?bD�Wɪ,       ���E	�d���z�A��*

A2S/average_reward_1�`D� U,       ���E	�dݞ�z�A��*

A2S/average_reward_1��]D�7/u,       ���E	X= ��z�A��*

A2S/average_reward_1
�[D�l�,       ���E	��^��z�A��*

A2S/average_reward_1�YD$��,       ���E	����z�A��*

A2S/average_reward_1�nWDU� �,       ���E	�g��z�A��*

A2S/average_reward_1�EUD��P\,       ���E	��%��z�A��*

A2S/average_reward_1�SD�Cܥ,       ���E	�h��z�A��*

A2S/average_reward_13�PDS3�,       ���E	e����z�A��*

A2S/average_reward_1��ND��p,       ���E	΋���z�A��*

A2S/average_reward_13�LD�<��,       ���E	��H��z�A��*

A2S/average_reward_1�{JD�#�,       ���E	 ���z�A��*

A2S/average_reward_1�RHD��r�,       ���E	��С�z�A��*

A2S/average_reward_1�'FDx��,       ���E	�v��z�A��*

A2S/average_reward_1� DD��<�,       ���E	"g��z�A��*

A2S/average_reward_1�AD�M�,       ���E	����z�A��*

A2S/average_reward_1��?DIx@F,       ���E	k=��z�A��*

A2S/average_reward_13�=D�^�,       ���E	��P��z�A��*

A2S/average_reward_1=Z;D��3m,       ���E	�����z�A��*

A2S/average_reward_1.9D�~�,       ���E	�ޣ�z�A��*

A2S/average_reward_1R7D�I��,       ���E	�#%��z�A��*

A2S/average_reward_1��4D~�x�,       ���E	�n��z�A��*

A2S/average_reward_1R�2DD<e�,       ���E	Pɲ��z�A��*

A2S/average_reward_1͌0DȉP,       ���E	Us���z�A��*

A2S/average_reward_1�`.DZ�,       ���E	��?��z�A��*

A2S/average_reward_1�:,DͫDR,       ���E	�{��z�A��*

A2S/average_reward_1�*D��.�,       ���E	�ֺ��z�A��*

A2S/average_reward_1=�'D\��e,       ���E	���z�A��*

A2S/average_reward_1H�%D-��,       ���E	�%L��z�A��*

A2S/average_reward_1ד#D���,       ���E	)H���z�A��*

A2S/average_reward_1�j!D�ػ�,       ���E	�V��z�A��*

A2S/average_reward_1HAD�j8n,       ���E	ҍ8��z�A��*

A2S/average_reward_1�D;~�X,       ���E	�D���z�A��*

A2S/average_reward_1 �DF
	G,       ���E	�Ş�z�A��*

A2S/average_reward_13�D^�$a,       ���E	�k��z�A��*

A2S/average_reward_1��D�vܞ,       ���E	�Z��z�A��*

A2S/average_reward_1qmD�zl�,       ���E	�ͨ��z�A��*

A2S/average_reward_1�CD]�.,       ���E	k��z�A��*

A2S/average_reward_1D��,       ���E	�1��z�A��*

A2S/average_reward_1��D��w�,       ���E	i�~��z�A܀*

A2S/average_reward_1��D{N��,       ���E	�>ɩ�z�A�*

A2S/average_reward_1�	D��g",       ���E	���z�A�*

A2S/average_reward_1fvDĚ��,       ���E	��Q��z�A��*

A2S/average_reward_1\OD�+�L,       ���E	�Օ��z�A��*

A2S/average_reward_1�'D>��],       ���E	����z�A��*

A2S/average_reward_1� DMO,       ���E	<�+��z�A��*

A2S/average_reward_1ͬ�C�y;,       ���E	�u��z�A��*

A2S/average_reward_1�Z�C9��,,       ���E	H���z�A��*

A2S/average_reward_1��C�VMX,       ���E	9���z�A��*

A2S/average_reward_13��C�r�,       ���E	͖0��z�A��*

A2S/average_reward_1�Z�C��@,       ���E	*�r��z�A��*

A2S/average_reward_1�CY�
,       ���E	(����z�A��*

A2S/average_reward_1)��CC��,       ���E	����z�AƎ*

A2S/average_reward_1�l�C�D3',       ���E	��0��z�Aɏ*

A2S/average_reward_1{�COM��,       ���E	@�z��z�A֐*

A2S/average_reward_1���C\;3�,       ���E	�ǭ�z�A�*

A2S/average_reward_1)|�C
�E�,       ���E	Am��z�A�*

A2S/average_reward_1f&�C��?`,       ���E	J��z�A�*

A2S/average_reward_1
��CO�,       ���E	~����z�A��*

A2S/average_reward_1 ��C��,       ���E	��Ԯ�z�A��*

A2S/average_reward_133�C�O�,       ���E	'��z�A��*

A2S/average_reward_1)ܼCA$��,       ���E	�vW��z�A��*

A2S/average_reward_1���C@3��,       ���E		���z�A��*

A2S/average_reward_1{4�CI���,       ���E	��ѯ�z�A��*

A2S/average_reward_1�گC~2/�,       ���E	-��z�A��*

A2S/average_reward_1H��C���,       ���E	zYm��z�A��*

A2S/average_reward_1\/�C`���,       ���E	n���z�A��*

A2S/average_reward_1�ڢC���u,       ���E	����z�A��*

A2S/average_reward_1���C*/`,       ���E	��@��z�A��*

A2S/average_reward_1�5�Cw7I,       ���E	�����z�A��*

A2S/average_reward_1��CyƟw,       ���E	�ֱ�z�Aˡ*

A2S/average_reward_1)��Cʖ�f,       ���E	���z�AТ*

A2S/average_reward_1fF�C�ٕe,       ���E	?�U��z�Aѣ*

A2S/average_reward_1��C��k�,       ���E	Kܜ��z�Aڤ*

A2S/average_reward_1ᚄC+?�,,       ���E	�e޲�z�A�*

A2S/average_reward_1=J�C"�i�,       ���E	�)'��z�A�*

A2S/average_reward_1q�wC�,       ���E	R�k��z�A��*

A2S/average_reward_1�QoC���,       ���E	�����z�A��*

A2S/average_reward_13�fCť#v,       ���E	�����z�A��*

A2S/average_reward_1{^C[-q�,       ���E	%M��z�A��*

A2S/average_reward_1RxUC��Z,       ���E	J����z�A��*

A2S/average_reward_1 �LC���K,       ���E	 
״�z�A��*

A2S/average_reward_1)DC��nT,       ���E	x���z�A��*

A2S/average_reward_1Rx;C���/,       ���E	+�X��z�A��*

A2S/average_reward_1��2Cܣơ,       ���E	�%���z�A��*

A2S/average_reward_1�:*C�؄A,       ���E	_�ڵ�z�A��*

A2S/average_reward_1��!C�)t�,       ���E	s��z�Aǲ*

A2S/average_reward_1��Cϵ�D,       ���E	��^��z�Aʳ*

A2S/average_reward_1 @CO;��,       ���E	�E���z�A״*

A2S/average_reward_1��C�~�@�       �v@�	<S׶�z�A״*�

A2S/klE�>

A2S/average_advantagex�:>

A2S/policy_network_lossf@p=

A2S/value_network_loss��@

A2S/q_network_loss]��@�b,       ���E	ᚣ��z�A��*

A2S/average_reward_1�QC��C,       ���E	@\���z�A��*

A2S/average_reward_1C�}aj,       ���E	�w}��z�A��*

A2S/average_reward_1f�!CA2��,       ���E	��a��z�A��*

A2S/average_reward_1 @*C�&d�,       ���E	=]F��z�A��*

A2S/average_reward_1��2C,�	�,       ���E	�\3��z�A��*

A2S/average_reward_1��;C�[,       ���E	���z�A��*

A2S/average_reward_1�+DC��F0,       ���E	,����z�A��*

A2S/average_reward_1��LC&n,       ���E	�����z�A��*

A2S/average_reward_1�hUC��,       ���E	|�	��z�A�*

A2S/average_reward_1�^C����,       ���E	�2��z�Aϊ*

A2S/average_reward_1��fC>�j�,       ���E	��(��z�A��*

A2S/average_reward_1=JoC�lN,       ���E	ϰ��z�A��*

A2S/average_reward_1H�wCä�,       ���E	�q*��z�A��*

A2S/average_reward_1fF�CѴ�,       ���E	3H!��z�A�*

A2S/average_reward_1���C���,       ���E	\���z�Aױ*

A2S/average_reward_1
��C6q�_,       ���E	����z�A��*

A2S/average_reward_1�H�C]]��,       ���E	�����z�A��*

A2S/average_reward_1q��C�l�,       ���E	O���z�A��*

A2S/average_reward_1�CD���,       ���E	0���z�A��*

A2S/average_reward_1HA�C9��,       ���E	�^*��z�A��*

A2S/average_reward_1��C��,       ���E	(w��z�A��*

A2S/average_reward_1qݢC�H,       ���E	����z�A��*

A2S/average_reward_1\/�C�%��,       ���E	�,���z�A��*

A2S/average_reward_1��C:],       ���E	�!���z�A��*

A2S/average_reward_13ӯC�p�,       ���E	&����z�A��*

A2S/average_reward_1=*�C���,       ���E	�c���z�Aχ*

A2S/average_reward_1q}�Cm\P,       ���E	�����z�A��*

A2S/average_reward_1μCq�,       ���E	.(���z�A��*

A2S/average_reward_1  �C���:,       ���E	�����z�A��*

A2S/average_reward_1Rx�C�d,       ���E	.���z�A�*

A2S/average_reward_1���C�0�~,       ���E	� ���z�A׮*

A2S/average_reward_13�C��	�,       ���E	�s���z�A��*

A2S/average_reward_1�c�CyO��,       ���E	�����z�A��*

A2S/average_reward_1��C3���,       ���E	�C���z�A��*

A2S/average_reward_13�Cˌ&,       ���E	����z�A��*

A2S/average_reward_1�^�C��&D,       ���E	�[���z�A��*

A2S/average_reward_13��C�ң,       ���E	u���z�A��*

A2S/average_reward_1  �C����,       ���E	��	�z�A��*

A2S/average_reward_1�Q�C��c:,       ���E	޳�z�A��*

A2S/average_reward_1ͬ�C�~o,       ���E	�6�z�A��*

A2S/average_reward_1���C7A�,       ���E	6~�	�z�A��*

A2S/average_reward_1�Q�C�
(u,       ���E	״�z�Aτ*

A2S/average_reward_1���C�ꌓ,       ���E	����z�A��*

A2S/average_reward_1=� D�%G,       ���E	m)��z�A��*

A2S/average_reward_1
'D����,       ���E	�}��z�A��*

A2S/average_reward_1�PD�{�B,       ���E	F���z�A�*

A2S/average_reward_1�|D}V,�,       ���E	����z�A׫*

A2S/average_reward_1f�	D_�I�,       ���E	�#��z�A��*

A2S/average_reward_1)�D$$��,       ���E	Rb��z�A��*

A2S/average_reward_1f�D��6�,       ���E	�;��z�A��*

A2S/average_reward_1� D�p��,       ���E	t���z�A��*

A2S/average_reward_1RHD���,       ���E	�^��z�A��*

A2S/average_reward_1�sD�(�H,       ���E	D��!�z�A��*

A2S/average_reward_1�DyB�,       ���E	�#�z�A��*

A2S/average_reward_1��Dg�#,       ���E	�&�z�A��*

A2S/average_reward_1)�D(�Jl,       ���E	\�'�z�A��*

A2S/average_reward_1�D�{N,       ���E	��)�z�A��*

A2S/average_reward_1�<D�%m�,       ���E	Ĳ�+�z�Aρ*

A2S/average_reward_1{d!D�䔹,       ���E	'�-�z�A��*

A2S/average_reward_1��#D�m,       ���E	\��/�z�A��*

A2S/average_reward_1ͼ%D?���,       ���E	M�1�z�A��*

A2S/average_reward_13�'D���,       ���E	l�3�z�A�*

A2S/average_reward_1)*D��e,       ���E	�ȿ5�z�Aר*

A2S/average_reward_1�3,D�jU,       ���E	X��7�z�A��*

A2S/average_reward_1 `.Dk�`,       ���E	�έ9�z�A��*

A2S/average_reward_1Å0D��,       ���E	Z�;�z�A��*

A2S/average_reward_1)�2D��~!,       ���E	�ַ=�z�A��*

A2S/average_reward_1
�4D� ��,       ���E	;>�?�z�A��*

A2S/average_reward_1��6D���m,       ���E	ٰA�z�A��*

A2S/average_reward_1=*9Dw�%,       ���E	��C�z�A��*

A2S/average_reward_1�P;D��Sz,       ���E	��E�z�A��*

A2S/average_reward_1)|=DD[�,       ���E	JѓG�z�A��*

A2S/average_reward_1{�?D(b��,       ���E	pk�I�z�A��*

A2S/average_reward_1 �AD��,       ���E	s��K�z�A��*

A2S/average_reward_1��CD��a�,       ���E	�n�M�z�A��*

A2S/average_reward_1�)FD)��,       ���E	+��O�z�A��*

A2S/average_reward_1�RHDk��9,       ���E	��|Q�z�A��*

A2S/average_reward_1�|JD�h�,       ���E	�4�S�z�A�*

A2S/average_reward_1f�LDemm,       ���E	��U�z�Aץ*

A2S/average_reward_1\�ND��x,       ���E	���W�z�A��*

A2S/average_reward_1R�PD*}y�,       ���E	���Y�z�A��*

A2S/average_reward_1)SD��l�,       ���E	�fj[�z�A��*

A2S/average_reward_1
GUDh���,       ���E	��]]�z�A��*

A2S/average_reward_1{tWD��/,       ���E	�/U_�z�A��*

A2S/average_reward_1͜YD�Tg�,       ���E	��Da�z�A��*

A2S/average_reward_1�[D��N�,       ���E	;�/c�z�A��*

A2S/average_reward_1��]DNm��,       ���E	m!e�z�A��*

A2S/average_reward_1�`D�S�,       ���E	�2/g�z�A��*

A2S/average_reward_1q=bD�m�,       ���E	�&i�z�A��*

A2S/average_reward_1edD�ܓ�,       ���E	 �$k�z�A��*

A2S/average_reward_1)�fD�;�,       ���E	�H)m�z�A��*

A2S/average_reward_1=�hDy��w,       ���E	�Mo�z�A��*

A2S/average_reward_13�jD/{/�,       ���E	�zq�z�A��*

A2S/average_reward_1)mD�}�A,       ���E	+ǵr�z�A�*

A2S/average_reward_1H1oD�!�,       ���E	�{�t�z�Aע*

A2S/average_reward_1�[qD��j�,       ���E	�v�z�A��*

A2S/average_reward_1׃sD��t,       ���E	5�x�z�A��*

A2S/average_reward_1�uDK3�,       ���E	6�`z�z�A��*

A2S/average_reward_1=�wD
H:�,       ���E	�T|�z�A��*

A2S/average_reward_1  zD�­�,       ���E	��e~�z�A��*

A2S/average_reward_1  zD���,       ���E	�Uy��z�A��*

A2S/average_reward_1  zD�dH�,       ���E	��w��z�A��*

A2S/average_reward_1  zD��Z,       ���E	-Wf��z�A��*

A2S/average_reward_1  zD%}�c,       ���E	��R��z�A��*

A2S/average_reward_1  zD��,       ���E	0��z�A��*

A2S/average_reward_1  zDJ81%,       ���E	��	��z�A��*

A2S/average_reward_1  zDK-sB,       ���E	�d
��z�A��*

A2S/average_reward_1  zD��d;,       ���E	U����z�A��*

A2S/average_reward_1  zD{3,       ���E	�,��z�A��*

A2S/average_reward_1  zDȭ�G�       �v@�	+�=��z�A��*�

A2S/kl��>

A2S/average_advantage���

A2S/policy_network_lossQ��

A2S/value_network_loss���C

A2S/q_network_loss�'�C�/��,       ���E	�4w��z�A��*

A2S/average_reward_1=�wDZ���,       ���E	����z�A�*

A2S/average_reward_1 �uD}�,       ���E	}Hޑ�z�AԒ*

A2S/average_reward_1UsD�ؽ~,       ���E	-���z�A��*

A2S/average_reward_1�qD�C,       ���E	H�J��z�A��*

A2S/average_reward_1q�nD)	G�,       ���E	A�~��z�A��*

A2S/average_reward_1ףlD%�d,       ���E	�.���z�A��*

A2S/average_reward_1�ajD:r�,       ���E	����z�A�*

A2S/average_reward_1=*hDX��,       ���E	�(��z�Aٗ*

A2S/average_reward_1��eD�?�\,       ���E	t�b��z�A��*

A2S/average_reward_1{�cD�G�J,       ���E	i왓�z�A��*

A2S/average_reward_1�waD���,       ���E	QXғ�z�A��*

A2S/average_reward_1�<_D㬕,       ���E	��z�A��*

A2S/average_reward_1H]Dt�^,       ���E	*�B��z�A�*

A2S/average_reward_1f�ZDW��,       ���E	\w��z�A֜*

A2S/average_reward_1��XD���,       ���E	:=���z�A��*

A2S/average_reward_1�KVD_��<,       ���E	wԔ�z�A��*

A2S/average_reward_1�
TDy���,       ���E	����z�A��*

A2S/average_reward_13�QD^/�,       ���E	��F��z�A��*

A2S/average_reward_1f�OD8��	,       ���E	Y^~��z�A�*

A2S/average_reward_1)\MD,k|�,       ���E	U6���z�Aա*

A2S/average_reward_1�"KD�f��,       ���E	���z�A��*

A2S/average_reward_1f�HD�5�,,       ���E	����z�A��*

A2S/average_reward_1�FD5B,       ���E	�W��z�A��*

A2S/average_reward_1qmDD���0,       ���E	4V���z�A��*

A2S/average_reward_1�3BDVCS�,       ���E	)Oʖ�z�A�*

A2S/average_reward_1R�?D�.�,       ���E	����z�Aئ*

A2S/average_reward_1�=D@HY*,       ���E	�D��z�A��*

A2S/average_reward_1��;D���,       ���E	�D}��z�A��*

A2S/average_reward_1�B9D*VA@,       ���E	�����z�A��*

A2S/average_reward_1f7D� ��,       ���E	����z�A��*

A2S/average_reward_1q�4D`�rP,       ���E	a���z�A�*

A2S/average_reward_1��2Dgxn,       ���E	S<��z�Aȫ*

A2S/average_reward_1qM0D2B,       ���E	V�s��z�A��*

A2S/average_reward_1
.D��{�,       ���E	?���z�A��*

A2S/average_reward_1\�+D�=�,       ���E	���z�A��*

A2S/average_reward_13�)D�!u�,       ���E	I�%��z�A��*

A2S/average_reward_1�j'DX"��,       ���E	��]��z�A�*

A2S/average_reward_1�.%Du'�,       ���E	�G���z�Aְ*

A2S/average_reward_1 �"D�o@,       ���E	��ə�z�A��*

A2S/average_reward_1H� D���,       ���E	2z���z�A��*

A2S/average_reward_1qmD�Ќ,       ���E	[A4��z�A��*

A2S/average_reward_1�,D�Fǁ,       ���E	V'k��z�A�*

A2S/average_reward_1��D��~�,       ���E	TϚ��z�Aд*

A2S/average_reward_1׳DF��j,       ���E	��Ԛ�z�AƵ*

A2S/average_reward_1\D��,       ���E	ʽ��z�A��*

A2S/average_reward_1�AD��$+,       ���E	&�;��z�A��*

A2S/average_reward_1RDU6�p,       ���E	!k��z�A��*

A2S/average_reward_1f�D�^�,       ���E	FZ���z�A�*

A2S/average_reward_1��D �$,       ���E	�!ԛ�z�Aɹ*

A2S/average_reward_1�H
D��Ԓ,       ���E	���z�A��*

A2S/average_reward_13D�G$k,       ���E	�R?��z�A��*

A2S/average_reward_1��DTy�A,       ���E	��t��z�A��*

A2S/average_reward_1{�DLt{},       ���E	q����z�A��*

A2S/average_reward_1)\D�QU�,       ���E	s`��z�A�*

A2S/average_reward_1�C�C���,       ���E	����z�Aо*

A2S/average_reward_1��C"�b,       ���E	7�I��z�A��*

A2S/average_reward_1\O�C:,²,       ���E	����z�A��*

A2S/average_reward_1H��C�4,       ���E	����z�A��*

A2S/average_reward_1�h�Cl@0^,       ���E	�1���z�A��*

A2S/average_reward_1)��C���,       ���E	��"��z�A��*

A2S/average_reward_1f��C�j] ,       ���E	9sZ��z�A��*

A2S/average_reward_1��C}Rg,       ���E	�����z�A��*

A2S/average_reward_1 ��CTtW�,       ���E	YVϞ�z�A��*

A2S/average_reward_1=*�C�� �,       ���E	����z�A��*

A2S/average_reward_1{��C����,       ���E	�B��z�A��*

A2S/average_reward_1�1�CF��d,       ���E	dP���z�A��*

A2S/average_reward_1���C��� ,       ���E	y絟�z�A��*

A2S/average_reward_1�B�C�c<+,       ���E	�T��z�A��*

A2S/average_reward_1�ȿCS�qA,       ���E	"x$��z�A��*

A2S/average_reward_1�P�C!xk	,       ���E	#b��z�A��*

A2S/average_reward_1qݶC�qѓ,       ���E	����z�A��*

A2S/average_reward_1Ha�C}g+,       ���E	'�à�z�A��*

A2S/average_reward_1�Cy��	,       ���E	( ��z�A��*

A2S/average_reward_1)|�C�p[�,       ���E	5�6��z�A��*

A2S/average_reward_1H�C�c�D,       ���E	V�i��z�A��*

A2S/average_reward_1\��Cꨧ�,       ���E	󀠡�z�A��*

A2S/average_reward_1)�Cv��k,       ���E	Y�ˡ�z�A��*

A2S/average_reward_1R��C8W�,       ���E	�e���z�A��*

A2S/average_reward_1�"�CS$��,       ���E	��1��z�A��*

A2S/average_reward_1\��C��`�,       ���E	�qb��z�A��*

A2S/average_reward_1�1�C�mC,       ���E	X����z�A��*

A2S/average_reward_1 ��C�Y*,       ���E	(֢�z�A��*

A2S/average_reward_1fF�C^i]l,       ���E	^���z�A��*

A2S/average_reward_1H�yC9���,       ���E	cW=��z�A��*

A2S/average_reward_1õpC���,       ���E	�j��z�A��*

A2S/average_reward_1�gC���|,       ���E	���z�A��*

A2S/average_reward_1f�^C�r�4,       ���E	��٣�z�A��*

A2S/average_reward_1�UC{$�U,       ���E	���z�A��*

A2S/average_reward_1MC���,       ���E	$G��z�A��*

A2S/average_reward_1  DC�\�U,       ���E	ze���z�A��*

A2S/average_reward_1);C*N��,       ���E	����z�A��*

A2S/average_reward_1R82C��o-,       ���E	m����z�A��*

A2S/average_reward_1Ha)C=cb,       ���E	f�6��z�A��*

A2S/average_reward_1q} CA'�,       ���E	��l��z�A��*

A2S/average_reward_1��CXJ��,       ���E	m����z�A��*

A2S/average_reward_1f�C�K1�,       ���E	j�٥�z�A��*

A2S/average_reward_1R�C4|J�,       ���E	;%
��z�A��*

A2S/average_reward_1ף�B�Q,       ���E	8��z�A��*

A2S/average_reward_13��B��A�,       ���E	�5r��z�A��*

A2S/average_reward_1���BQ~Z�,       ���E	\���z�A��*

A2S/average_reward_1���B,��/,       ���E	Cۦ�z�A��*

A2S/average_reward_1���B��qW,       ���E	A���z�A��*

A2S/average_reward_1���B�~�,       ���E	�4H��z�A��*

A2S/average_reward_1���B:ۢ�,       ���E	"w��z�A��*

A2S/average_reward_1���B�0��,       ���E	R����z�A��*

A2S/average_reward_1�B��,       ���E	=����z�A��*

A2S/average_reward_1�(�B�Ѳa,       ���E	X2��z�A��*

A2S/average_reward_1���B1�r�,       ���E	q\ ��z�A��*

A2S/average_reward_1���B<�|2,       ���E	'�Q��z�A��*

A2S/average_reward_1{��B���       �v@�	�P���z�A��*�

A2S/kl�t'>

A2S/average_advantage�*�

A2S/policy_network_lossR1�=

A2S/value_network_loss]A

A2S/q_network_loss�21A�Z�a,       ���E	�z��z�A��*

A2S/average_reward_1�z�B�~o�,       ���E	�6n��z�A��*

A2S/average_reward_1�Q�B����,       ���E	.܊��z�A̂*

A2S/average_reward_1
�C��D�,       ���E	U�z��z�A��*

A2S/average_reward_1��C�i�L,       ���E	�>���z�AȊ*

A2S/average_reward_1��CP�:�,       ���E	l����z�A��*

A2S/average_reward_1H�C��B�,       ���E	�w��z�A��*

A2S/average_reward_1ףC���,       ���E	3����z�A��*

A2S/average_reward_1õC.C2,       ���E	�I���z�A��*

A2S/average_reward_1��CG��,       ���E	4g���z�A՚*

A2S/average_reward_1��C�?uI,       ���E	�J���z�A�*

A2S/average_reward_1�CyYv ,       ���E	�̈��z�AТ*

A2S/average_reward_1�$C|ve,       ���E	t��z�A��*

A2S/average_reward_1�-C�HfU,       ���E	�'f��z�A��*

A2S/average_reward_1f�5Cm&�,       ���E	��q��z�A��*

A2S/average_reward_15C�/�L,       ���E	n-��z�Aβ*

A2S/average_reward_1�04C5b,       ���E	u���z�A��*

A2S/average_reward_1�=C�w��,       ���E	gS���z�A��*

A2S/average_reward_1\FCdtZ,       ���E	�(���z�A��*

A2S/average_reward_1�QECP ��,       ���E	<�s��z�A��*

A2S/average_reward_1�BNCM��',       ���E	��j��z�A��*

A2S/average_reward_1f&WC���,       ���E	ei��z�A��*

A2S/average_reward_1H!`CN�),       ���E	z}��z�A��*

A2S/average_reward_1f&iC-��},       ���E	�ц��z�A��*

A2S/average_reward_1�5hC�2m,       ���E	3.O��z�A��*

A2S/average_reward_1{qC�!�,       ���E	�A��z�A��*

A2S/average_reward_1zC�.D�,       ���E	v:9��z�A��*

A2S/average_reward_13s�C���,       ���E	��C��z�A��*

A2S/average_reward_1��C��'<,       ���E	��*��z�A��*

A2S/average_reward_1�h�C6���,       ���E	�D���z�A��*

A2S/average_reward_1f�C&��x,       ���E	�B���z�AȘ*

A2S/average_reward_1n�CGh��,       ���E	g����z�Aޘ*

A2S/average_reward_1��C�j�,       ���E	\�G��z�AƠ*

A2S/average_reward_1׃�CN��,       ���E	�D��z�A��*

A2S/average_reward_1q��C ���,       ���E	;���z�A��*

A2S/average_reward_1ff�C�n�,       ���E	h=%��z�A��*

A2S/average_reward_1q��C1��,       ���E	!��z�A��*

A2S/average_reward_1�p�Cqs��,       ���E	zB��z�A��*

A2S/average_reward_1{��C����,       ���E	����z�A��*

A2S/average_reward_1n�C*���,       ���E	�"���z�A��*

A2S/average_reward_1\�C�
l,       ���E	v���z�A��*

A2S/average_reward_1�Z�C^�e,       ���E	�h���z�A��*

A2S/average_reward_1���C*��,       ���E	����z�A��*

A2S/average_reward_1
w�Cc�%�,       ���E	�J���z�A��*

A2S/average_reward_1��C;�),       ���E	@���z�A��*

A2S/average_reward_1)\�C�^@�,       ���E	�����z�A��*

A2S/average_reward_1���C�*�,       ���E	%���z�A��*

A2S/average_reward_1�p�CD���,       ���E	�����z�A��*

A2S/average_reward_1���CW$S,       ���E	����z�A��*

A2S/average_reward_1��C~{%B,       ���E	����z�A��*

A2S/average_reward_1���C���%,       ���E	�?��z�A��*

A2S/average_reward_1͌�C\�q6,       ���E	<���z�A��*

A2S/average_reward_1��C��,       ���E	�+��z�A��*

A2S/average_reward_1��C��e�,       ���E	|[$��z�A��*

A2S/average_reward_1)�C�,       ���E	��-��z�A��*

A2S/average_reward_1��C.�J�,       ���E	���z�A��*

A2S/average_reward_1{�C�X�,       ���E	����z�A��*

A2S/average_reward_1׃�C�hB�,       ���E	'����z�Aަ*

A2S/average_reward_1��C.H�5,       ���E	(���z�AƮ*

A2S/average_reward_1q}�C��,       ���E	����z�A��*

A2S/average_reward_1���C���g,       ���E	����z�AǶ*

A2S/average_reward_1���C	�%o,       ���E	����z�A��*

A2S/average_reward_1�C� �,       ���E	s&���z�A��*

A2S/average_reward_1Rx�C3J�,       ���E	�m� �z�A��*

A2S/average_reward_1=��C�x,8,       ���E	R��z�A��*

A2S/average_reward_1e�C����,       ���E	'���z�A��*

A2S/average_reward_1���C}N�,       ���E	����z�A��*

A2S/average_reward_1��C��1�,       ���E	Q<��z�A��*

A2S/average_reward_1��C�	{�,       ���E	#��z�A��*

A2S/average_reward_1\?D�}�,       ���E	��z�A��*

A2S/average_reward_1�xD>��,       ���E	��
�z�A��*

A2S/average_reward_1��D=1��,       ���E	�h�
�z�A��*

A2S/average_reward_1�|Dv�-,       ���E	2��z�A��*

A2S/average_reward_1fFD}�ӷ,       ���E	��z�A��*

A2S/average_reward_1HD��yX,       ���E	y��z�A��*

A2S/average_reward_1)LD���-,       ���E	2�z�A��*

A2S/average_reward_1D����,       ���E	;l�z�A��*

A2S/average_reward_1��D�ʇ,       ���E	"��z�A�*

A2S/average_reward_1�	D��V,       ���E	�N�z�A׍*

A2S/average_reward_13SDEvG�,       ���E	�Z��z�A��*

A2S/average_reward_1{�D/wM�,       ���E	#���z�A��*

A2S/average_reward_1q�D�4tB,       ���E	6(�z�A��*

A2S/average_reward_13�D�rn4,       ���E	z"�z�A��*

A2S/average_reward_1��D���d,       ���E	K;�z�A��*

A2S/average_reward_1�D�ltJ,       ���E	�{�z�A�*

A2S/average_reward_1\?D���,       ���E	c/%�z�A��*

A2S/average_reward_1�D���J,       ���E	Rb,�z�A��*

A2S/average_reward_1{DD�q��,       ���E	@��z�A��*

A2S/average_reward_1~D@v@�,       ���E	s�!�z�A��*

A2S/average_reward_1)�Dh�	q,       ���E	��#�z�A��*

A2S/average_reward_13�D(�,       ���E	;a+#�z�A��*

A2S/average_reward_1 �DXg+,       ���E	�+%�z�A��*

A2S/average_reward_1�� D��,       ���E	]D'�z�A��*

A2S/average_reward_1�:#Duh�,       ���E	ٚ<)�z�A��*

A2S/average_reward_1u%D��e�,       ���E	DWP+�z�A��*

A2S/average_reward_1��'D����,       ���E	@�^+�z�A��*

A2S/average_reward_1��'D���l,       ���E	�>-�z�A��*

A2S/average_reward_1)�)DY��v,       ���E	J}$/�z�Aу*

A2S/average_reward_1�*,D9v8,       ���E	@U1�z�A��*

A2S/average_reward_1n.Dʋ�N,       ���E	 (�2�z�A��*

A2S/average_reward_1)�0D}�p,       ���E	3��4�z�A��*

A2S/average_reward_1)�0DE$,,       ���E	|��6�z�A�*

A2S/average_reward_1)�0D�:�),       ���E	]�8�z�A٪*

A2S/average_reward_1)�0D����,       ���E	5�:�z�A��*

A2S/average_reward_1)�0D��<,       ���E	�J�<�z�A��*

A2S/average_reward_1\3D�̿�,       ���E	�<�z�A��*

A2S/average_reward_1ͬ0D���,       ���E	�>�z�A��*

A2S/average_reward_1ͬ0DZ��,       ���E	f��@�z�A��*

A2S/average_reward_1  3DDz:,       ���E	e��B�z�A��*

A2S/average_reward_1��5D���,       ���E	���D�z�A��*

A2S/average_reward_1�8D�͌�,       ���E	��F�z�A��*

A2S/average_reward_1�y:D;Lj�,       ���E	���H�z�A��*

A2S/average_reward_1�y:D6��,       ���E	xa�J�z�A��*

A2S/average_reward_1�y:D���n,       ���E	���L�z�A��*

A2S/average_reward_1�y:D��Ox,       ���E	h)�L�z�A��*

A2S/average_reward_1Rx:Dʁ��,       ���E	��N�z�A�� *

A2S/average_reward_1��<DYFt�,       ���E	55�P�z�A� *

A2S/average_reward_1��<D���,       ���E	F��R�z�Aʐ *

A2S/average_reward_1��<D����,       ���E	5�T�z�A�� *

A2S/average_reward_1fV?D�D,       ���E	�9xV�z�A�� *

A2S/average_reward_1fV?D�[�T�       �v@�	�:vW�z�A�� *�

A2S/klԽK>

A2S/average_advantage��#?

A2S/policy_network_loss��=

A2S/value_network_loss��C

A2S/q_network_loss���Cj�},       ���E	���W�z�A�� *

A2S/average_reward_1��<D/��,       ���E	��W�z�Aؠ *

A2S/average_reward_1~:D�E�,       ���E	@'�W�z�A�� *

A2S/average_reward_1H18D2���,       ���E	���W�z�Aǡ *

A2S/average_reward_1�78D�
�,       ���E	���W�z�A� *

A2S/average_reward_1��5D�K׼,       ���E	��W�z�A�� *

A2S/average_reward_1�a3DfTŖ,       ���E	ֆ�W�z�A�� *

A2S/average_reward_1H�0D��K.,       ���E	�
X�z�A�� *

A2S/average_reward_1�.D/���,       ���E	�fUX�z�Aܣ *

A2S/average_reward_1�h,D�H�^,       ���E	HdX�z�A�� *

A2S/average_reward_1��)D��U�,       ���E	]�wX�z�A�� *

A2S/average_reward_1H�'D�!^,       ���E	�ԇX�z�A�� *

A2S/average_reward_1
�'D-ć�,       ���E	���X�z�A� *

A2S/average_reward_1�%D�:5%,       ���E	�01Y�z�A� *

A2S/average_reward_1�Z#DZ�s�,       ���E	��wY�z�A� *

A2S/average_reward_1�.!D���,       ���E	�1�Y�z�A�� *

A2S/average_reward_1��!D�eA|,       ���E	�1�Y�z�A�� *

A2S/average_reward_1
'DQm,�,       ���E	�W.Z�z�A̪ *

A2S/average_reward_1  DG���,       ���E	�2?Z�z�A� *

A2S/average_reward_13�D��,       ���E	�%PZ�z�A�� *

A2S/average_reward_1�%D�|7F,       ���E	�+�Z�z�A�� *

A2S/average_reward_1q�D����,       ���E	�^�Z�z�A�� *

A2S/average_reward_1�
DI>ӷ,       ���E	��[�z�A� *

A2S/average_reward_13�D��?,       ���E	�[�z�A�� *

A2S/average_reward_1��Dr#�,       ���E	r�.[�z�A�� *

A2S/average_reward_1� D�J��,       ���E	\(A[�z�Aͮ *

A2S/average_reward_1��D����,       ���E	��[�z�A֯ *

A2S/average_reward_1� D�/�,       ���E	�[�z�A�� *

A2S/average_reward_1�D_n�_,       ���E	���[�z�A�� *

A2S/average_reward_1�D���v,       ���E	��[�z�A�� *

A2S/average_reward_1��
D�4P�,       ���E	���[�z�A� *

A2S/average_reward_1��
DeZ�9,       ���E	4�[�z�A�� *

A2S/average_reward_1
WD��,       ���E	��
\�z�A�� *

A2S/average_reward_1�^DrO�p,       ���E	��F\�z�A�� *

A2S/average_reward_1��D1Fo,       ���E	e�U\�z�A˲ *

A2S/average_reward_1),D�X/�,       ���E	�?t\�z�A�� *

A2S/average_reward_1��D�W��,       ���E	%��\�z�A�� *

A2S/average_reward_1�kDà8�,       ���E	��\�z�A̳ *

A2S/average_reward_1q��C_C,       ���E	��\�z�A� *

A2S/average_reward_1�,�Cq�K�,       ���E	��\�z�A�� *

A2S/average_reward_1�Y�Cj��,       ���E	��\�z�A�� *

A2S/average_reward_13s�CF޻�,       ���E	���\�z�A�� *

A2S/average_reward_1��C�>"�,       ���E	h�]�z�A�� *

A2S/average_reward_1��C|T5�,       ���E	B]�z�A۵ *

A2S/average_reward_1R8�C�G��,       ���E	lF+]�z�A�� *

A2S/average_reward_1Ha�CXW3�,       ���E	�{:]�z�A�� *

A2S/average_reward_1�l�C\�o,       ���E	B�J]�z�A�� *

A2S/average_reward_1)|�C킲�,       ���E	���]�z�Aͷ *

A2S/average_reward_133�Cd�,       ���E	���]�z�A� *

A2S/average_reward_1q]�C��_N,       ���E	�]�z�A�� *

A2S/average_reward_1Õ�CF��h,       ���E	�U�]�z�A�� *

A2S/average_reward_1 ��C_	,,       ���E	z�]�z�Aܸ *

A2S/average_reward_1���C�Q(m,       ���E	X�"^�z�A�� *

A2S/average_reward_1�~�C1�k,       ���E	��3^�z�A�� *

A2S/average_reward_1=��C!�W,       ���E	�M}^�z�A�� *

A2S/average_reward_1�B�C�m/�,       ���E	�z�^�z�A� *

A2S/average_reward_1�+�C��#�,       ���E	&�_�z�A� *

A2S/average_reward_1���C�ɖ�,       ���E	M�%_�z�A�� *

A2S/average_reward_1R��C�"S,       ���E	�|f_�z�A�� *

A2S/average_reward_1��C����,       ���E	E�t_�z�A�� *

A2S/average_reward_1\��C����,       ���E	ؤ�_�z�A޿ *

A2S/average_reward_1 �C��+y,       ���E	!��_�z�A�� *

A2S/average_reward_13�C�K�,       ���E	��_�z�A�� *

A2S/average_reward_1=*�C��e,       ���E	�_�z�A�� *

A2S/average_reward_1\O�CcG�9,       ���E	e�_�z�A�� *

A2S/average_reward_1{t�C��",       ���E	{��_�z�A�� *

A2S/average_reward_1�~�C$_X,       ���E	�`�z�A�� *

A2S/average_reward_1f�C�6	D,       ���E	��+`�z�A�� *

A2S/average_reward_1\/�Cr��h,       ���E	�?`�z�A�� *

A2S/average_reward_1)\�CΞ��,       ���E	�V�`�z�A�� *

A2S/average_reward_1R�C:a-�,       ���E	u!�`�z�A�� *

A2S/average_reward_1H!�C�M�p,       ���E	���`�z�A�� *

A2S/average_reward_1���C`OB7,       ���E	I--a�z�A�� *

A2S/average_reward_1Rx�Cݱ&5,       ���E	\�@a�z�A�� *

A2S/average_reward_1ף�CV��,       ���E	m�a�z�A�� *

A2S/average_reward_1E�CE�� ,       ���E	���a�z�A�� *

A2S/average_reward_1׃�C?�M�,       ���E	T׶a�z�A�� *

A2S/average_reward_1ͬ�C�Z�,       ���E	�a�z�A�� *

A2S/average_reward_13�{C�]H,       ���E	c��a�z�A�� *

A2S/average_reward_1rC�Ɏ�,       ���E	T��a�z�A�� *

A2S/average_reward_1�LhC��s8,       ���E	�%*b�z�A�� *

A2S/average_reward_1f�_C���,       ���E	�?8b�z�A�� *

A2S/average_reward_1R�UCH�C?,       ���E	zEb�z�A�� *

A2S/average_reward_1=JLC�>��,       ���E	.#Ub�z�A�� *

A2S/average_reward_1ףBC呂�,       ���E	�{bb�z�A�� *

A2S/average_reward_1��8CVN�,       ���E	n8tb�z�A�� *

A2S/average_reward_1H!9C�X.�,       ���E	�c�b�z�A�� *

A2S/average_reward_1�Y/C��,       ���E	P�b�z�A�� *

A2S/average_reward_1�%C|��>,       ���E	���b�z�A�� *

A2S/average_reward_1C�-�},       ���E	���b�z�A�� *

A2S/average_reward_1{TCg2�,       ���E	�b�z�A�� *

A2S/average_reward_1��C�y��,       ���E	�?�b�z�A�� *

A2S/average_reward_1���B��Y	,       ���E	���b�z�A�� *

A2S/average_reward_1\��BM}�Z,       ���E	-�c�z�A�� *

A2S/average_reward_1�u�B���r,       ���E	Z.c�z�A�� *

A2S/average_reward_13��B,�y�,       ���E	�?c�z�A�� *

A2S/average_reward_133�Bh�	�,       ���E	*�Kc�z�A�� *

A2S/average_reward_1�̱BF�ޚ,       ���E	�[c�z�A�� *

A2S/average_reward_1�z�B�@�,       ���E	��ic�z�A�� *

A2S/average_reward_1{�BPP,       ���E	�	xc�z�A�� *

A2S/average_reward_1ffoBz��=,       ���E	��c�z�A�� *

A2S/average_reward_1�GoB0L�H,       ���E	���c�z�A�� *

A2S/average_reward_1q=oBc�7�,       ���E	�Y�c�z�A�� *

A2S/average_reward_1q=mB�#(,       ���E	�D�c�z�A�� *

A2S/average_reward_1ffmB��	,       ���E	WGd�z�A�� *

A2S/average_reward_1��rB��,       ���E	��d�z�A�� *

A2S/average_reward_1��rB�q-�,       ���E	��%d�z�A�� *

A2S/average_reward_1�(sB5�uw,       ���E	�J:d�z�A�� *

A2S/average_reward_1�GsB��*�,       ���E	U@Ld�z�A�� *

A2S/average_reward_1q=nB�Eq,       ���E	ƶ�d�z�A�� *

A2S/average_reward_1��rB�==,       ���E	sơd�z�A�� *

A2S/average_reward_1ףrBr�e,       ���E	���d�z�A�� *

A2S/average_reward_1\�rB9tK�,       ���E	�ٽd�z�A�� *

A2S/average_reward_1�mBzcV�,       ���E	J�d�z�A�� *

A2S/average_reward_1H�hBa��,       ���E	"[�d�z�A�� *

A2S/average_reward_1�eB��A,       ���E	���d�z�A�� *

A2S/average_reward_1ff_B8��,       ���E	l�d�z�A�� *

A2S/average_reward_1)\_BdSi�,       ���E	�e�z�A�� *

A2S/average_reward_1�G[B���,       ���E	�A"e�z�A�� *

A2S/average_reward_1�p[B5�,       ���E	1�7e�z�A�� *

A2S/average_reward_1��[Bh�       �v@�	?C~e�z�A�� *�

A2S/klT>

A2S/average_advantage18>

A2S/policy_network_loss�

A2S/value_network_loss�l�B

A2S/q_network_loss*��BF<��,       ���E	!D�e�z�A�� *

A2S/average_reward_1�WB=�e,       ���E	��e�z�A�� *

A2S/average_reward_1{VB���E,       ���E	/�e�z�A�� *

A2S/average_reward_1�zPB���o,       ���E	o�e�z�A�� *

A2S/average_reward_1�OBJ4��,       ���E	|ҝe�z�A�� *

A2S/average_reward_1��NB�b�Y,       ���E		��e�z�A�� *

A2S/average_reward_1�QNB*�rU,       ���E	;>�e�z�A�� *

A2S/average_reward_1�GIB_���,       ���E	���e�z�A�� *

A2S/average_reward_1=
HBn+L�,       ���E	G��e�z�A�� *

A2S/average_reward_1��GB�W�,       ���E	�e�z�A�� *

A2S/average_reward_1\�FB j� ,       ���E	� �e�z�A�� *

A2S/average_reward_1��EB���,       ���E	.��e�z�A�� *

A2S/average_reward_1ףDB��,       ���E	V��e�z�A�� *

A2S/average_reward_1=
DB7���,       ���E	�`�e�z�A�� *

A2S/average_reward_1��@B��g,       ���E	#e�e�z�A�� *

A2S/average_reward_1ff@B���|,       ���E	�e�e�z�A�� *

A2S/average_reward_133>B���,       ���E	�4�e�z�A�� *

A2S/average_reward_1�Q=Bm^�/,       ���E	t�f�z�A�� *

A2S/average_reward_1�=B��,       ���E	�zf�z�A�� *

A2S/average_reward_1=
<B�<�,       ���E	M0f�z�A�� *

A2S/average_reward_1�(;BH4�C,       ���E	�if�z�A�� *

A2S/average_reward_1q=:B�R��,       ���E	�Df�z�A�� *

A2S/average_reward_1��7B�B�,       ���E	Em$f�z�A�� *

A2S/average_reward_1�6Bw�x�,       ���E	5q4f�z�A�� *

A2S/average_reward_1\�6BAn,       ���E	u�Df�z�A�� *

A2S/average_reward_1\�6Bi8��,       ���E	�5Kf�z�A�� *

A2S/average_reward_1��5B����,       ���E	!}[f�z�A�� *

A2S/average_reward_1��5B�>��,       ���E	8�ff�z�A�� *

A2S/average_reward_1{1B|��,       ���E	�lf�z�A�� *

A2S/average_reward_1�G0Bj�h,       ���E	#�wf�z�A�� *

A2S/average_reward_1�z/B��,       ���E	��|f�z�A�� *

A2S/average_reward_1�.B�g�j,       ���E	Ԍf�z�A�� *

A2S/average_reward_1�G.B�-(�,       ���E	�h�f�z�A�� *

A2S/average_reward_1�p(B9c�,       ���E	W,�f�z�A�� *

A2S/average_reward_133(B����,       ���E	|8�f�z�A�� *

A2S/average_reward_1��"B6$�,       ���E	���f�z�A�� *

A2S/average_reward_1�Bnk��,       ���E	��f�z�A�� *

A2S/average_reward_1)\B�,       ���E	��f�z�A�� *

A2S/average_reward_1\�B+V+�,       ���E	~��f�z�A�� *

A2S/average_reward_1H�BA�`�,       ���E	'��f�z�A�� *

A2S/average_reward_1  B%�G,       ���E	H��f�z�A�� *

A2S/average_reward_1�QBG�,       ���E	���f�z�A�� *

A2S/average_reward_133B��Ug,       ���E	���f�z�A�� *

A2S/average_reward_1=
B��o,       ���E	  �f�z�A�� *

A2S/average_reward_133B��Tj,       ���E	���f�z�A�� *

A2S/average_reward_1)\B5�,       ���E	��f�z�A�� *

A2S/average_reward_1�B�4Ǥ,       ���E	A�g�z�A�� *

A2S/average_reward_1��	B�z��,       ���E	�g�z�A�� *

A2S/average_reward_1�B!��,       ���E	�\g�z�A�� *

A2S/average_reward_1��B)a,       ���E	�'g�z�A�� *

A2S/average_reward_1)\B�^�>,       ���E	��g�z�A�� *

A2S/average_reward_1�pB��$,       ���E	h)g�z�A�� *

A2S/average_reward_1�(�AWNb,       ���E	Ȧ,g�z�A�� *

A2S/average_reward_1  �A��؉,       ���E	}�1g�z�A�� *

A2S/average_reward_1  �A����,       ���E	U�:g�z�A�� *

A2S/average_reward_1�G�A�Y�,       ���E	�Eg�z�A�� *

A2S/average_reward_1=
�A�g�6,       ���E	�Og�z�A�� *

A2S/average_reward_1  �A�G�,       ���E	��Zg�z�A�� *

A2S/average_reward_1=
�A�)��,       ���E	o0eg�z�A�� *

A2S/average_reward_1�z�A�k��,       ���E	r_sg�z�A�� *

A2S/average_reward_1H��Aǖ,       ���E	Jwg�z�A�� *

A2S/average_reward_1���Ar�K�,       ���E	%{g�z�A�� *

A2S/average_reward_1���Avh�,       ���E	]�g�z�A�� *

A2S/average_reward_133�AX��,       ���E	@�g�z�A�� *

A2S/average_reward_1���AS�7,       ���E	{��g�z�A�� *

A2S/average_reward_1�p�A�g�,       ���E	y��g�z�A�� *

A2S/average_reward_1��A�,��,       ���E	��g�z�A�� *

A2S/average_reward_1�(�Az8',       ���E	���g�z�A�� *

A2S/average_reward_1���AC�oN,       ���E	B��g�z�A�� *

A2S/average_reward_1  �A$��H,       ���E	$�g�z�A�� *

A2S/average_reward_1R��A�>�,       ���E	]4�g�z�A�� *

A2S/average_reward_1��An�(,       ���E	/@�g�z�A�� *

A2S/average_reward_1q=�A�x��,       ���E	�o�g�z�A�� *

A2S/average_reward_1ff�AӦ,,       ���E	�w�g�z�A�� *

A2S/average_reward_133�A@h��,       ���E	St�g�z�A�� *

A2S/average_reward_1)\�A�>9,       ���E	�y�g�z�A�� *

A2S/average_reward_1�Q�A(QFP,       ���E	���g�z�A�� *

A2S/average_reward_1=
�A���,       ���E	�Th�z�A�� *

A2S/average_reward_1{�A�V,�,       ���E	?h�z�A�� *

A2S/average_reward_1R��A�*,       ���E	�h�z�A�� *

A2S/average_reward_1���Aq��,       ���E	��h�z�A�� *

A2S/average_reward_1��A�T,       ���E	��h�z�A�� *

A2S/average_reward_1R��AC�=,       ���E	ۃ%h�z�A�� *

A2S/average_reward_133�A��h ,       ���E	��1h�z�A�� *

A2S/average_reward_1�Q�A�;��,       ���E	�=h�z�A�� *

A2S/average_reward_1R��A��,       ���E	�Qh�z�A�� *

A2S/average_reward_1���A�w"�,       ���E	\�Wh�z�A�� *

A2S/average_reward_1�G�A�7��,       ���E	�[h�z�A�� *

A2S/average_reward_1�G�Az��,       ���E	�_h�z�A�� *

A2S/average_reward_1��A�6�_,       ���E	��fh�z�A�� *

A2S/average_reward_1�G�AY�\�,       ���E	��nh�z�A�� *

A2S/average_reward_1�z�A�yI�,       ���E	�rh�z�A�� *

A2S/average_reward_1���A(E�,       ���E	�1wh�z�A�� *

A2S/average_reward_1)\�A��g�,       ���E	��h�z�A�� *

A2S/average_reward_1)\�A=�,       ���E	֥�h�z�A�� *

A2S/average_reward_1�G�A��",       ���E	���h�z�A�� *

A2S/average_reward_1�z�A�]�,       ���E	�ˠh�z�A�� *

A2S/average_reward_1�Q�A�1c�,       ���E	�$�h�z�A�� *

A2S/average_reward_133�AE�d�,       ���E	��h�z�A�� *

A2S/average_reward_1��AX���,       ���E	���h�z�A�� *

A2S/average_reward_1H�A@z�,       ���E	9�h�z�A�� *

A2S/average_reward_1  �A���,       ���E	uc�h�z�A�� *

A2S/average_reward_1���A��kW,       ���E	J��h�z�A�� *

A2S/average_reward_1=
�AAJ��,       ���E	���h�z�A�� *

A2S/average_reward_1ף�A�͎�,       ���E	�w�h�z�A�� *

A2S/average_reward_1�Q�Ae�M_,       ���E	d�h�z�A�� *

A2S/average_reward_1\��AS�h,       ���E	`��h�z�A�� *

A2S/average_reward_1
׍A��Z,       ���E	��h�z�A�� *

A2S/average_reward_1{�A�i��,       ���E	�# i�z�A�� *

A2S/average_reward_133�Ade�,       ���E	7fi�z�A�� *

A2S/average_reward_1)\�A1��,       ���E	3�
i�z�A�� *

A2S/average_reward_133�AL�$,       ���E	Qi�z�A�� *

A2S/average_reward_1q=�A��Cy,       ���E	nLi�z�A�� *

A2S/average_reward_1  �A���,       ���E	a�(i�z�A�� *

A2S/average_reward_1q=�A��H�,       ���E	��9i�z�A�� *

A2S/average_reward_1)\�A��E,       ���E	�>i�z�A�� *

A2S/average_reward_1��A���,       ���E	�4Hi�z�A�� *

A2S/average_reward_1  �A���C,       ���E	+�Pi�z�A�� *

A2S/average_reward_1�p�A�Y(T,       ���E	�4Xi�z�A�� *

A2S/average_reward_1��A�GZ,       ���E	[`i�z�A�� *

A2S/average_reward_1
׍A?f�J,       ���E	bhgi�z�A�� *

A2S/average_reward_133�AWLL,       ���E	t�mi�z�A�� *

A2S/average_reward_1��A��T�,       ���E	��ri�z�A�� *

A2S/average_reward_1\��A�FG�,       ���E	׾�i�z�A�� *

A2S/average_reward_1\��Ag�4,       ���E	��i�z�A�� *

A2S/average_reward_1���AR�kK,       ���E	'�i�z�A�� *

A2S/average_reward_1=
�ApP�7,       ���E	h �i�z�A�� *

A2S/average_reward_1�G�ATt��,       ���E	��i�z�A�� *

A2S/average_reward_1
׉A��,       ���E	���i�z�A�� *

A2S/average_reward_1�G�Aabpq,       ���E	M��i�z�A�� *

A2S/average_reward_1  �A�]8ח       �v@�	'j�z�A�� *�

A2S/kl��>

A2S/average_advantage��	�

A2S/policy_network_loss`�

A2S/value_network_loss�4A

A2S/q_network_loss�z\Awn��,       ���E	�cj�z�A�� *

A2S/average_reward_1���A���',       ���E	n�xl�z�A�� *

A2S/average_reward_1���A�
Y~,       ���E	N�jn�z�A�� *

A2S/average_reward_1=
B��,       ���E	��sp�z�A��!*

A2S/average_reward_1��BB�'o,       ���E	��br�z�A��!*

A2S/average_reward_1\�jBxxq�,       ���E	�at�z�Aߐ!*

A2S/average_reward_1{�B�_X,       ���E	��ov�z�Aǘ!*

A2S/average_reward_1�p�BVQ"�,       ���E	�RYx�z�A��!*

A2S/average_reward_1R8�B�3f%,       ���E	ެ�x�z�A��!*

A2S/average_reward_13��B��GS,       ���E	���z�z�A��!*

A2S/average_reward_1�z�B����,       ���E	P�|�z�A�!*

A2S/average_reward_1\�B���,       ���E	͖�~�z�Aٹ!*

A2S/average_reward_1�k�Bby,       ���E	3ѥ��z�A��!*

A2S/average_reward_1�C��a,       ���E	�/���z�A��!*

A2S/average_reward_1q}C«�,       ���E	�����z�A��!*

A2S/average_reward_1�hC�F��,       ���E	��|��z�A��!*

A2S/average_reward_1�:C���,       ���E	��R��z�A��!*

A2S/average_reward_1��(C�b<,       ���E	2V��z�A��!*

A2S/average_reward_1��2C��-Q,       ���E	�'T��z�A��!*

A2S/average_reward_1��<CHPL,       ���E	��M��z�A��!*

A2S/average_reward_1)�FC��8�,       ���E	e�+��z�A��"*

A2S/average_reward_1��PC�e\u,       ���E	��)��z�A�"*

A2S/average_reward_1EZC���~,       ���E	|&��z�Aя"*

A2S/average_reward_1�0dCWd��,       ���E	���z�A��"*

A2S/average_reward_1�nC[�5N,       ���E	�]��z�A��"*

A2S/average_reward_1�wC_�/�,       ���E	�ш��z�A��"*

A2S/average_reward_1{zC�j�,       ���E	����z�A��"*

A2S/average_reward_1��C���j,       ���E	䓜�z�Aݰ"*

A2S/average_reward_1{ԆC.y�5,       ���E	��f��z�AŸ"*

A2S/average_reward_1{��C�C��,       ���E	FZ���z�A��"*

A2S/average_reward_1=��CP��c,       ���E	��m��z�A��"*

A2S/average_reward_1�~�C�s�,       ���E	%�w��z�A��"*

A2S/average_reward_13s�C:�T�,       ���E	PO\��z�A��"*

A2S/average_reward_1ff�C���,       ���E	��U��z�A��"*

A2S/average_reward_1N�C�á�,       ���E	�G��z�A��"*

A2S/average_reward_1�,�C���6,       ���E	3�G��z�A��"*

A2S/average_reward_1H!�C�6��,       ���E	J�V��z�A��"*

A2S/average_reward_1{�C��[�,       ���E	�gL��z�A��"*

A2S/average_reward_1��C%��),       ���E	|"F��z�AՆ#*

A2S/average_reward_1H�C��v,       ���E	��a��z�A��#*

A2S/average_reward_1��C=Y��,       ���E	5\7��z�A��#*

A2S/average_reward_1{��CN��W,       ���E	đ���z�A�#*

A2S/average_reward_1{��C�R+$,       ���E	c9n��z�A˟#*

A2S/average_reward_1�l�Cd���,       ���E	̓d��z�A��#*

A2S/average_reward_1fF�C�$�,       ���E	
BC��z�A��#*

A2S/average_reward_1�:�C�J,       ���E	
p��z�A��#*

A2S/average_reward_1�+�C����,       ���E	����z�A�#*

A2S/average_reward_1��CW���,       ���E	�f ��z�A��#*

A2S/average_reward_1q��C��h�,       ���E	iU��z�A��#*

A2S/average_reward_1���C^�L�,       ���E	6��z�A��#*

A2S/average_reward_1H��Ch7�N,       ���E	�?��z�A��#*

A2S/average_reward_1���Ca��,       ���E	����z�A��#*

A2S/average_reward_1���CB"�?,       ���E	�]��z�A��#*

A2S/average_reward_1���Cxl��,       ���E	:�]��z�A��#*

A2S/average_reward_1 ��C���%,       ���E	�E��z�A��#*

A2S/average_reward_1��D�Mb�,       ���E	��>��z�A��#*

A2S/average_reward_1�*D��F�,       ���E	t>)��z�A�$*

A2S/average_reward_1��D�t�,       ���E	qc.��z�Aю$*

A2S/average_reward_1�	D^}��,       ���E	�
%��z�A��$*

A2S/average_reward_1��D���1,       ���E	��'��z�A��$*

A2S/average_reward_1�D�_�>,       ���E	)M��z�A��$*

A2S/average_reward_1H�D^��q,       ���E	uU
��z�A�$*

A2S/average_reward_1��D쇍,       ���E	����z�Aٵ$*

A2S/average_reward_1�sD){�,       ���E	3���z�A��$*

A2S/average_reward_1��D[�9F,       ���E	i�
��z�A��$*

A2S/average_reward_1�WD:�?�,       ���E	gi��z�A��$*

A2S/average_reward_1��D�E_�,       ���E	����z�A��$*

A2S/average_reward_15D1n{,       ���E	����z�A��$*

A2S/average_reward_1f�!D��G@,       ���E	�#���z�A��$*

A2S/average_reward_1H!$D��,       ���E	.^���z�A��$*

A2S/average_reward_1�&D�pє,       ���E	^����z�A��$*

A2S/average_reward_1�)D, ,       ���E	ͱ���z�A��$*

A2S/average_reward_1H�+DVȅt,       ���E	K���z�A�%*

A2S/average_reward_1��-D��P�,       ���E	/�D��z�AÅ%*

A2S/average_reward_1=j.D���,       ���E	��3��z�A��%*

A2S/average_reward_1��0Dٕ�L,       ���E	p�"��z�A��%*

A2S/average_reward_1^3DV��,       ���E	����z�A��%*

A2S/average_reward_1��5D����,       ���E	�[���z�A�%*

A2S/average_reward_1{D8D�'�t,       ���E	�����z�Aˬ%*

A2S/average_reward_1\�:D��,       ���E	����z�A��%*

A2S/average_reward_1R8=DI#�
,       ���E	A����z�A��%*

A2S/average_reward_1��?D0I�),       ���E	����z�A��%*

A2S/average_reward_1�#BDN��S,       ���E	���z�A��%*

A2S/average_reward_1=�DD�fy,       ���E	�	��z�A��%*

A2S/average_reward_1fGD�%�R,       ���E	bȠ�z�A��%*

A2S/average_reward_1 pID*2�,       ���E	G�	�z�A��%*

A2S/average_reward_1��KDo$�',       ���E	�̎�z�A��%*

A2S/average_reward_1�\ND�ߘ�,       ���E	 ��z�A��%*

A2S/average_reward_1H�PD�I1:,       ���E	��z�A��%*

A2S/average_reward_1�HSD�U:J,       ���E	Rȧ�z�AÂ&*

A2S/average_reward_1\�UD�&"�,       ���E	>e��z�A��&*

A2S/average_reward_1f6XD<I�,       ���E	<��z�A��&*

A2S/average_reward_1��ZD/Tv,       ���E	�#l�z�A��&*

A2S/average_reward_1R(]D�;G�,       ���E	[�F�z�A�&*

A2S/average_reward_1ד_D?��,       ���E	3�z�A˩&*

A2S/average_reward_1)bDO�-,       ���E	d��z�A��&*

A2S/average_reward_13�dD��/@,       ���E	��z�A��&*

A2S/average_reward_1��fDpj,       ���E		�� �z�A��&*

A2S/average_reward_1ffiD<x:,       ���E	�=�"�z�A��&*

A2S/average_reward_1��kD-�@+,       ���E	l��$�z�A��&*

A2S/average_reward_1q=nDQ�Ћ,       ���E	>��&�z�A��&*

A2S/average_reward_1 @pDRJ�,       ���E	�]�(�z�A��&*

A2S/average_reward_1 @pD*�,       ���E	v~B)�z�A��&*

A2S/average_reward_1�NnD=�M�,       ���E	B3+�z�A��&*

A2S/average_reward_1�NnD�:��,       ���E	T�C-�z�A��&*

A2S/average_reward_1�NnD<Ќ�,       ���E	a_0/�z�A��&*

A2S/average_reward_1�NnD]㣮,       ���E	E,1�z�A��'*

A2S/average_reward_1�NnDI���,       ���E	]>'3�z�A��'*

A2S/average_reward_1�NnD�F��,       ���E	��95�z�A�'*

A2S/average_reward_1�3pD�^,       ���E	6�<7�z�Aژ'*

A2S/average_reward_1�3pD@x�,       ���E	:�=9�z�A '*

A2S/average_reward_1�3pD髧,       ���E	��/;�z�A��'*

A2S/average_reward_1�3pDh�.,       ���E	��;�z�A��'*

A2S/average_reward_1�@nD�_/,       ���E	]��=�z�A�'*

A2S/average_reward_1�@nD]�ޑ,       ���E	��|?�z�Aֹ'*

A2S/average_reward_1�@nD��bN,       ���E	oA�z�A��'*

A2S/average_reward_1�@nDF��q,       ���E	l<dC�z�A��'*

A2S/average_reward_1�@nDZ��,       ���E	�\,E�z�A��'*

A2S/average_reward_1�@nD.Z,       ���E	��F�z�A��'*

A2S/average_reward_1�@nD�;��,       ���E	h�AH�z�A��'*

A2S/average_reward_1�@nD���,       ���E	&�/J�z�A��'*

A2S/average_reward_1�@nD�Y��,       ���E	T�L�z�A��'*

A2S/average_reward_1�@nD�N�N,       ���E	m4'N�z�A��'*

A2S/average_reward_1�@nD.%?P,       ���E	4,9P�z�A��'*

A2S/average_reward_1�@nDb�R�,       ���E	?��P�z�AƁ(*

A2S/average_reward_1�@lD���,       ���E	K�R�z�A��(*

A2S/average_reward_1�)nD�LQ,       ���E	SӔT�z�A��(*

A2S/average_reward_1�)nD���,       ���E	z�qV�z�A��(*

A2S/average_reward_1�)nD�_��,       ���E	��vX�z�A�(*

A2S/average_reward_1�)nDe��,       ���E	�
eZ�z�AΨ(*

A2S/average_reward_1�)nDw#g�,       ���E	�vu\�z�A��(*

A2S/average_reward_1�)nDb�J�,       ���E	J�^�z�A��(*

A2S/average_reward_1�)nDT�,       ���E	��F`�z�A��(*

A2S/average_reward_1�)nD�qc�,       ���E	��b�z�A��(*

A2S/average_reward_1�)nDI��v,       ���E	�d�z�A��(*

A2S/average_reward_1�)nD�t�,       ���E	Z�e�z�A��(*

A2S/average_reward_1�)nD	ª9,       ���E	��g�z�A��(*

A2S/average_reward_1�)nD�S��,       ���E	lڌh�z�A��(*

A2S/average_reward_1�GlD�r�+,       ���E	�ęj�z�A��(*

A2S/average_reward_1�GlD觯,       ���E	�|l�z�A��(*

A2S/average_reward_1�GlD�@���       �v@�	�x�m�z�A��(*�

A2S/klэ�>

A2S/average_advantageGU>�

A2S/policy_network_lossT��

A2S/value_network_loss��C

A2S/q_network_loss�͸CҾe,       ���E	.�o�z�A��(*

A2S/average_reward_1f�kD��c2,       ���E	f1�p�z�A��(*

A2S/average_reward_1�lD�m�,       ���E	,er�z�A��)*

A2S/average_reward_1�lD/DP�,       ���E	Js�z�A��)*

A2S/average_reward_1\kD��9,       ���E	�?^s�z�A��)*

A2S/average_reward_1�iD�w?W,       ���E	�t�z�A��)*

A2S/average_reward_1
WgD�Ҿ�,       ���E	[St�z�A��)*

A2S/average_reward_1�8eD�,�,       ���E	�@v�z�A��)*

A2S/average_reward_1�8eD�!Ph,       ���E	�ǫv�z�Aɔ)*

A2S/average_reward_1)<cD���o,       ���E	{_~w�z�A��)*

A2S/average_reward_1��aD�UY|,       ���E	Z6x�z�A�)*

A2S/average_reward_1�>`D��h,       ���E	Fzz�z�Aբ)*

A2S/average_reward_1�5bDB�F,       ���E	N��z�z�A��)*

A2S/average_reward_1 �`D�C��,       ���E	�?<|�z�AԪ)*

A2S/average_reward_1{�_D"��d,       ���E	��F}�z�A�)*

A2S/average_reward_1ד^D��<�,       ���E	L��}�z�A�)*

A2S/average_reward_1ͼ\D���,       ���E	���~�z�A��)*

A2S/average_reward_1 �[D����,       ���E	�1��z�A��)*

A2S/average_reward_1��ZD��/G,       ���E	6P���z�AƼ)*

A2S/average_reward_1�XD�P�,       ���E	"����z�Aڿ)*

A2S/average_reward_1��WD��	,       ���E	Z*9��z�A��)*

A2S/average_reward_1ͼUDΚ�a,       ���E	�cу�z�A��)*

A2S/average_reward_1�KUD����,       ���E	��#��z�A��)*

A2S/average_reward_1)\TD��,       ���E	?K��z�A��)*

A2S/average_reward_1�7SD�l�,       ���E	p����z�A��)*

A2S/average_reward_1R(QD1�+,       ���E	�����z�A��)*

A2S/average_reward_1R(QD���H,       ���E	q����z�A��)*

A2S/average_reward_1ODŇ�,       ���E	dҊ�z�A��)*

A2S/average_reward_1��NDS�I0,       ���E	cό�z�A��)*

A2S/average_reward_1��ND��,       ���E	H]Ȏ�z�A��)*

A2S/average_reward_1��NDX
>�,       ���E	��]��z�A��)*

A2S/average_reward_1�MD���,       ���E	��e��z�A��)*

A2S/average_reward_1�MD�b�,       ���E	Z^4��z�A��**

A2S/average_reward_1�KD3�_�,       ���E	튒�z�Aׁ**

A2S/average_reward_1�KDݬ�i,       ���E	@����z�A��**

A2S/average_reward_1�KD�r��,       ���E	�ˎ��z�A��**

A2S/average_reward_1RHJD���,       ���E	�Oi��z�A��**

A2S/average_reward_1HJDu^��,       ���E	��'��z�A��**

A2S/average_reward_1�qHD�H,       ���E	����z�A�**

A2S/average_reward_1�jHDJ��,       ���E	SV���z�A��**

A2S/average_reward_1��FD�}��,       ���E	�ߊ��z�Aߤ**

A2S/average_reward_1=JEDգ�3,       ���E	uΛ�z�A�**

A2S/average_reward_1{$CD���,       ���E	g�
��z�A�**

A2S/average_reward_1=�@D��,       ���E	﷯��z�A��**

A2S/average_reward_1RH?D~��2,       ���E	C�w��z�A��**

A2S/average_reward_1��=D- ,       ���E	f�ʝ�z�A�**

A2S/average_reward_13�;D�4V,       ���E	Y�_��z�A��**

A2S/average_reward_1��9Dr�q*,       ���E	e�(��z�AԶ**

A2S/average_reward_1Å9D&��5,       ���E	�.��z�A��**

A2S/average_reward_1Å9D/���,       ���E	�.7��z�A��**

A2S/average_reward_1Å9D7��,       ���E	7���z�A��**

A2S/average_reward_13�7D�y>,       ���E	��W��z�A��**

A2S/average_reward_1��5D�j
�,       ���E	����z�A��**

A2S/average_reward_1{d4D+-eB,       ���E	�̦�z�A��**

A2S/average_reward_1�2D��,       ���E	YC���z�A��**

A2S/average_reward_1
71D�@,       ���E	&�0��z�A��**

A2S/average_reward_1��/Dt�,       ���E	���z�A��**

A2S/average_reward_1)L/D�b��,       ���E	fW��z�A��**

A2S/average_reward_1�3.D��,       ���E	��߫�z�A��**

A2S/average_reward_1H�,D��"�,       ���E	����z�A��**

A2S/average_reward_1�,D����,       ���E	���z�A��**

A2S/average_reward_1��+D.r�,       ���E	v郯�z�A��**

A2S/average_reward_1=�)DLVg�,       ���E	4��z�A��**

A2S/average_reward_1�Y+D���[,       ���E	�W���z�A��**

A2S/average_reward_1=J)D}>�,       ���E	˃��z�A��**

A2S/average_reward_1{t'D����,       ���E	�����z�A��**

A2S/average_reward_1�%Dn��|,       ���E	,�+��z�A�+*

A2S/average_reward_1{�#D߫y�,       ���E	�����z�A��+*

A2S/average_reward_1)�!D�qP,       ���E	��z��z�A��+*

A2S/average_reward_1�` DB��,       ���E	K�ʴ�z�A��+*

A2S/average_reward_1�BDФ�,       ���E	�D��z�A��+*

A2S/average_reward_1�1D�a�,       ���E	�A���z�A��+*

A2S/average_reward_1f�D6�Xk,       ���E	#.��z�Aߐ+*

A2S/average_reward_1�D��q,       ���E	�����z�A��+*

A2S/average_reward_1�DD�,       ���E	��A��z�A��+*

A2S/average_reward_1)<DR��,       ���E	~ȸ�z�Aŗ+*

A2S/average_reward_1�jD=�޼,       ���E	����z�AŚ+*

A2S/average_reward_1��D�hy,       ���E	޳ܹ�z�A�+*

A2S/average_reward_1�D��%,       ���E	�.���z�Aʣ+*

A2S/average_reward_1�DA�r�,       ���E	�޼�z�A��+*

A2S/average_reward_1��D�q��,       ���E	ΕP��z�A��+*

A2S/average_reward_1H�D9�,       ���E	!"ݽ�z�A��+*

A2S/average_reward_1qD��J,       ���E	�˭��z�Aί+*

A2S/average_reward_1��DR��M,       ���E	����z�A��+*

A2S/average_reward_1��D�,       ���E	,?N��z�A��+*

A2S/average_reward_1 �D}�u�,       ���E	�P��z�A̽+*

A2S/average_reward_1R�D�_��,       ���E	ty���z�A��+*

A2S/average_reward_1{D���,       ���E	����z�A��+*

A2S/average_reward_1��	D��($,       ���E	��v��z�A��+*

A2S/average_reward_13CDG�ּ,       ���E	G�_��z�A��+*

A2S/average_reward_1��D��PG,       ���E	xM��z�A��+*

A2S/average_reward_1��Dh��,       ���E	��O��z�A��+*

A2S/average_reward_1��D����,       ���E	�����z�A��+*

A2S/average_reward_1��D�f:,       ���E	�S���z�A��+*

A2S/average_reward_1q�D`���,       ���E	�X��z�A��+*

A2S/average_reward_1f�D�Ic�,       ���E	�T&��z�A��+*

A2S/average_reward_1�W Dު^P,       ���E	z���z�A��+*

A2S/average_reward_1�W DA�ID,       ���E	����z�A��+*

A2S/average_reward_1�9DrY,       ���E	R#v��z�A��+*

A2S/average_reward_1�9 DD��l,       ���E	��0��z�A��,*

A2S/average_reward_1
 D�.:�,       ���E	�����z�A��,*

A2S/average_reward_1�Z�C2���,       ���E	�VS��z�A��,*

A2S/average_reward_1��Cvuh�,       ���E	D���z�A؈,*

A2S/average_reward_1=*�C���,       ���E	����z�A֊,*

A2S/average_reward_1f��C$|��,       ���E	�����z�Aڏ,*

A2S/average_reward_1��CH8*,       ���E	ؗ���z�A��,*

A2S/average_reward_1��C�Go,       ���E	ʡ���z�A��,*

A2S/average_reward_1=j�CTc��,       ���E	}�U��z�A��,*

A2S/average_reward_1�g�C����,       ���E	h���z�A��,*

A2S/average_reward_1{��C����,       ���E	6ߛ��z�A��,*

A2S/average_reward_1���C��2,       ���E	è���z�A�,*

A2S/average_reward_1��C�VM,,       ���E	�Dm��z�AƩ,*

A2S/average_reward_1=*�C��#�,       ���E	�AI��z�A�,*

A2S/average_reward_1�'�CT��l,       ���E	�?+��z�A��,*

A2S/average_reward_1���C� މ,       ���E	"�*��z�AƸ,*

A2S/average_reward_1��C	��l,       ���E	,����z�Aϻ,*

A2S/average_reward_1ͬ�C��,       ���E	��9��z�A�,*

A2S/average_reward_1 ��C�9��,       ���E	F����z�A��,*

A2S/average_reward_1 ��C�7S�,       ���E	8�S��z�A��,*

A2S/average_reward_1�Z�C`Ę�,       ���E	^O��z�A��,*

A2S/average_reward_1�U�C� ��,       ���E	�H���z�A��,*

A2S/average_reward_133�C�}�,       ���E	~5���z�A��,*

A2S/average_reward_1��CJSq�,       ���E	Ux��z�A��,*

A2S/average_reward_13��CK`�,       ���E	֒���z�A��,*

A2S/average_reward_1�Y�C���,       ���E	��q��z�A��,*

A2S/average_reward_1���C��9/,       ���E	����z�A��,*

A2S/average_reward_1�h�CO���,       ���E	-�[��z�A��,*

A2S/average_reward_1��C�Y,       ���E	�#��z�A��,*

A2S/average_reward_1���C�6F�,       ���E	�
��z�A��,*

A2S/average_reward_1ͬ�C��� ,       ���E	�b'��z�A��,*

A2S/average_reward_1
w�C��',       ���E	�:r��z�A��,*

A2S/average_reward_1���CC;�Q,       ���E	��n��z�A��,*

A2S/average_reward_1���C�R�,       ���E	�����z�A��,*

A2S/average_reward_1�u�C��e�,       ���E	�Ω��z�A��,*

A2S/average_reward_1��C��,       ���E	3���z�A��-*

A2S/average_reward_1 ��C� �`,       ���E	ײ���z�A��-*

A2S/average_reward_1)��C�m,       ���E	�����z�Aօ-*

A2S/average_reward_1�Q�C�nb�,       ���E	٩��z�A��-*

A2S/average_reward_1���C�0��,       ���E	�{��z�A��-*

A2S/average_reward_13��C����,       ���E	�$~��z�Aѐ-*

A2S/average_reward_1��C���,       ���E	*�Q��z�A�-*

A2S/average_reward_1���C�Cj,       ���E	����z�A�-*

A2S/average_reward_1�'�C���,       ���E	�-���z�AӞ-*

A2S/average_reward_1)|�CyW��,       ���E	u�/��z�A�-*

A2S/average_reward_1��C�� �,       ���E	�����z�A��-*

A2S/average_reward_1���C�^�,       ���E	�t���z�A��-*

A2S/average_reward_1���C�s�r,       ���E	�����z�Aӭ-*

A2S/average_reward_1�0�Cl �M,       ���E	��+��z�A��-*

A2S/average_reward_1��CO},       ���E	L�z�A��-*

A2S/average_reward_1��CI���,       ���E	����z�A��-*

A2S/average_reward_1%�CI"_��       �v@�	���z�A��-*�

A2S/kl�ܺ>

A2S/average_advantageT�b�

A2S/policy_network_loss����

A2S/value_network_loss̰�B

A2S/q_network_loss��Bg�R,       ���E	���z�A��-*

A2S/average_reward_1=��C���",       ���E	�8��z�A��-*

A2S/average_reward_1�L�C���,       ���E	.
��z�A��-*

A2S/average_reward_1�g�C+h��,       ���E	�
�z�A��-*

A2S/average_reward_1{��Cc���,       ���E	���z�A��-*

A2S/average_reward_1���CJW,       ���E	�D��z�A��-*

A2S/average_reward_1�D7��G,       ���E	�΅�z�A��-*

A2S/average_reward_1)LD#��K,       ���E	_�~�z�A��-*

A2S/average_reward_1{dD�W��,       ���E	����z�Aـ.*

A2S/average_reward_1
�D���~,       ���E	���z�A��.*

A2S/average_reward_1=
DӐ6q,       ���E	*���z�A��.*

A2S/average_reward_1�D+�v�,       ���E	�×�z�A��.*

A2S/average_reward_1�D3%J,       ���E	Fo��z�A��.*

A2S/average_reward_1 0D0�5�,       ���E	��r�z�A�.*

A2S/average_reward_1\?	D�dr,       ���E	5�� �z�Aɯ.*

A2S/average_reward_1D1z��,       ���E	˪�"�z�A��.*

A2S/average_reward_1��D�R,       ���E	<��$�z�A��.*

A2S/average_reward_1�D8p�J,       ���E	K�}&�z�A��.*

A2S/average_reward_1q�D=5~@,       ���E	��_(�z�A��.*

A2S/average_reward_1�(Dڅ�H,       ���E	{�D*�z�A��.*

A2S/average_reward_1
GD��s,       ���E	~.,�z�A��.*

A2S/average_reward_1�WD�҇�,       ���E	��.�z�A��.*

A2S/average_reward_13D��V�,       ���E	k�&0�z�A��.*

A2S/average_reward_1��D�vD,       ���E	��2�z�A��.*

A2S/average_reward_1��D��-,       ���E	 �4�z�A��.*

A2S/average_reward_1�@Dn��d,       ���E	�y	6�z�A��/*

A2S/average_reward_1�DPD�,       ���E	�-�7�z�A��/*

A2S/average_reward_1)�D7g�,       ���E	YS�9�z�A��/*

A2S/average_reward_1��!D:���,       ���E	���;�z�A��/*

A2S/average_reward_1��!D8�+�,       ���E	cȱ=�z�A�/*

A2S/average_reward_1��"D3�,       ���E	v��?�z�Aɬ/*

A2S/average_reward_1��$D̮Sg,       ���E	���A�z�A��/*

A2S/average_reward_1\_&D4F�r,       ���E	5��C�z�A��/*

A2S/average_reward_1�'D�SJ�,       ���E	ћ�E�z�A��/*

A2S/average_reward_1�'D�`c�,       ���E	ɾ�G�z�A��/*

A2S/average_reward_1ͼ)D*(�,       ���E	�I�z�A��/*

A2S/average_reward_1{�*D���-,       ���E	��QK�z�A��/*

A2S/average_reward_1Rx+D�eB,       ���E	gCM�z�A��/*

A2S/average_reward_1�,DC�j,       ���E	�QO�z�A��/*

A2S/average_reward_1�9.D���,       ���E	��;Q�z�A��/*

A2S/average_reward_1ף/D��:,       ���E	�S�z�A��/*

A2S/average_reward_1=�/DvC.9,       ���E	�U�z�A��0*

A2S/average_reward_1=�/D�$�p,       ���E	���V�z�A��0*

A2S/average_reward_1�1D�M�,       ���E	�Y�z�A��0*

A2S/average_reward_1\�2D����,       ���E	�u	[�z�A��0*

A2S/average_reward_1f�4D�ҏ�,       ���E	ZH�\�z�A�0*

A2S/average_reward_1%6D��T�,       ���E	��^�z�Aɩ0*

A2S/average_reward_1%6D��en,       ���E	Y�`�z�A��0*

A2S/average_reward_1%6Dۣ�,       ���E	��b�z�A��0*

A2S/average_reward_1%8D:�44,       ���E	�^�d�z�A��0*

A2S/average_reward_1�W8D>!N,       ���E	M:�f�z�A��0*

A2S/average_reward_1�:D�	h,       ���E	�^|h�z�A��0*

A2S/average_reward_1 �;Dtp��,       ���E	�&�j�z�A��0*

A2S/average_reward_1�\=D���,       ���E	D�l�z�A��0*

A2S/average_reward_1=:?D�9�,       ���E	�en�z�A��0*

A2S/average_reward_1@D�89J,       ���E	ݎDp�z�A��0*

A2S/average_reward_1�|AD��0,       ���E	�'2r�z�A��0*

A2S/average_reward_1��BD���,       ���E	�?t�z�A��0*

A2S/average_reward_1 @CDx^&1,       ���E	Q�Lv�z�A��1*

A2S/average_reward_1f&EDъ{{,       ���E	�jNx�z�A��1*

A2S/average_reward_1{dED�,��,       ���E	�;z�z�A��1*

A2S/average_reward_1�iGD*�,       ���E	�LB|�z�A�1*

A2S/average_reward_1�XID���,       ���E	><.~�z�Aɦ1*

A2S/average_reward_1 �JD?,       ���E	n~<��z�A��1*

A2S/average_reward_1��JD�]��,       ���E	g�(��z�A��1*

A2S/average_reward_1)LD6c�r,       ���E	��$��z�A��1*

A2S/average_reward_1��MDaP�,       ���E	�M��z�A��1*

A2S/average_reward_1׳OD(�,       ���E	1$,��z�A��1*

A2S/average_reward_1 �QD�{B�,       ���E	�A��z�A��1*

A2S/average_reward_1R�QD��3�,       ���E	˨*��z�A��1*

A2S/average_reward_1R�QD[<u,       ���E	)���z�A��1*

A2S/average_reward_1q�SD'Y��,       ���E	��#��z�A��1*

A2S/average_reward_1q�SD�&�N,       ���E	9�/��z�A��1*

A2S/average_reward_1^UD���,       ���E	��9��z�A��1*

A2S/average_reward_1\?WD���|,       ���E	X�/��z�A��2*

A2S/average_reward_1�%YDyK4�,       ���E	n~4��z�A��2*

A2S/average_reward_1
G[D��m�,       ���E	�=��z�A��2*

A2S/average_reward_1q�\D�"	�,       ���E	�0I��z�A�2*

A2S/average_reward_1
w^D�օ�,       ���E	��O��z�Aɣ2*

A2S/average_reward_1��_DdŦ,       ���E	�W��z�A��2*

A2S/average_reward_1��`D4X��,       ���E	��K��z�A��2*

A2S/average_reward_1HcD�5�W,       ���E	<K;��z�A��2*

A2S/average_reward_1HcDά�,       ���E	��G��z�A��2*

A2S/average_reward_1�cDR��,       ���E	i�0��z�A��2*

A2S/average_reward_1\/eD��|/,       ���E	g5��z�A��2*

A2S/average_reward_1f&gD�1�},       ���E	EL��z�A��2*

A2S/average_reward_1�iDD�cE,       ���E	�-���z�A��2*

A2S/average_reward_1^jD��,       ���E	c@��z�A��2*

A2S/average_reward_1^jDW��Y,       ���E	@�ı�z�A��2*

A2S/average_reward_1�clD�1[�,       ���E	S���z�A��2*

A2S/average_reward_1)\nD�Y(,       ���E	�H���z�A��3*

A2S/average_reward_1H�oD*wk,       ���E	�����z�A��3*

A2S/average_reward_1 `qDVc�,       ���E	'���z�A��3*

A2S/average_reward_1 `qD�s��,       ���E	�����z�A�3*

A2S/average_reward_1q}sD��s,       ���E	c����z�Aɠ3*

A2S/average_reward_1׃uD�A,       ���E	��s��z�A��3*

A2S/average_reward_1׃uDl�%�,       ���E	��}��z�A��3*

A2S/average_reward_13�vDt�;�,       ���E	lƕ��z�A��3*

A2S/average_reward_1��xD��Ѭ,       ���E	�;s��z�A�3*

A2S/average_reward_1��xDz`�M,       ���E	��}��z�A��3*

A2S/average_reward_1  zDDϥ,       ���E	��k��z�A��3*

A2S/average_reward_1  zD�-,       ���E	i�<��z�A��3*

A2S/average_reward_1  zD��m,       ���E	^&A��z�A��3*

A2S/average_reward_1  zDFj,       ���E	y14��z�A��3*

A2S/average_reward_1  zD�6��,       ���E	�m5��z�A��3*

A2S/average_reward_1  zD�b?(,       ���E	C�9��z�A��3*

A2S/average_reward_1  zDґ��,       ���E	� ��z�A��3*

A2S/average_reward_1  zD�먎,       ���E	��8��z�A��4*

A2S/average_reward_1  zD��v,       ���E	E�:��z�A��4*

A2S/average_reward_1  zDM�MH,       ���E	��)��z�A�4*

A2S/average_reward_1  zD�X��,       ���E	�y��z�Aɝ4*

A2S/average_reward_1  zD��@,       ���E	�>��z�A��4*

A2S/average_reward_1  zD��5�,       ���E	�����z�A��4*

A2S/average_reward_1  zD�}s,       ���E	 z���z�A��4*

A2S/average_reward_1  zD=�C�,       ���E	j����z�A�4*

A2S/average_reward_1  zD�Մ,       ���E	c���z�A��4*

A2S/average_reward_1  zD��F,       ���E	3���z�A��4*

A2S/average_reward_1  zD��=,       ���E	T���z�A��4*

A2S/average_reward_1  zDH�4�,       ���E	 ����z�A��4*

A2S/average_reward_1  zDo7�,       ���E	5r���z�A��4*

A2S/average_reward_1  zD���,       ���E	D����z�A��4*

A2S/average_reward_1  zDP���,       ���E	K5���z�A��4*

A2S/average_reward_1  zD�2�_,       ���E	�5���z�A��4*

A2S/average_reward_1  zDBsX,       ���E	�����z�A��5*

A2S/average_reward_1  zDSO{s,       ���E	�����z�A��5*

A2S/average_reward_1  zD��I,       ���E	Z����z�A�5*

A2S/average_reward_1  zD�p;�,       ���E	{v��z�Aɚ5*

A2S/average_reward_1  zD��;,       ���E	oߖ��z�A��5*

A2S/average_reward_1  zD��� ,       ���E	sw� �z�A��5*

A2S/average_reward_1  zDsS��,       ���E	���z�A��5*

A2S/average_reward_1  zD ߨ�,       ���E	,>t�z�A�5*

A2S/average_reward_1  zD��@�,       ���E	D�s�z�A��5*

A2S/average_reward_1  zD�Ş�,       ���E	y�c�z�A��5*

A2S/average_reward_1  zD^9AA,       ���E	z�k
�z�A��5*

A2S/average_reward_1  zD	Ή,       ���E	��v�z�A��5*

A2S/average_reward_1  zDoPo�,       ���E	(�j�z�A��5*

A2S/average_reward_1  zD��=r,       ���E	U�z�z�A��5*

A2S/average_reward_1  zDX�|2,       ���E	���z�A��5*

A2S/average_reward_1  zD���,       ���E	��r�z�A��5*

A2S/average_reward_1  zDD���,       ���E	��E�z�A��6*

A2S/average_reward_1  zD�XH,       ���E	�PW�z�A��6*

A2S/average_reward_1  zDyuǛ,       ���E	�4N�z�A�6*

A2S/average_reward_1  zD���N,       ���E	7�=�z�Aɗ6*

A2S/average_reward_1  zD�ͳ,       ���E	@�-�z�A��6*

A2S/average_reward_1  zD���,       ���E	79* �z�A��6*

A2S/average_reward_1  zD�
�,       ���E	;`."�z�A��6*

A2S/average_reward_1  zD�'��,       ���E	ǒE$�z�A�6*

A2S/average_reward_1  zD�I�u,       ���E	9 &�z�AѾ6*

A2S/average_reward_1  zD��,       ���E	�n(�z�A��6*

A2S/average_reward_1  zD�W5�,       ���E	��*�z�A��6*

A2S/average_reward_1  zD�s�L�       �v@�	�"�+�z�A��6*�

A2S/kl|��>

A2S/average_advantage���

A2S/policy_network_loss���

A2S/value_network_loss� �C

A2S/q_network_loss['�C���,       ���E	�ʯ-�z�A��6*

A2S/average_reward_1  zD���,       ���E	���/�z�A��6*

A2S/average_reward_1  zD��o,       ���E	RW�1�z�A��6*

A2S/average_reward_1  zDS[4
,       ���E	�5�3�z�A��6*

A2S/average_reward_1  zDv"uM,       ���E	Z��5�z�A��6*

A2S/average_reward_1  zD��8,       ���E	]�7�z�A��6*

A2S/average_reward_1  zDس0�,       ���E	�kk9�z�A��7*

A2S/average_reward_1  zD�H��,       ���E	nsl;�z�A�7*

A2S/average_reward_1  zD�2�%,       ���E	�e@=�z�Aɔ7*

A2S/average_reward_1  zD�T�O,       ���E	�<E?�z�A��7*

A2S/average_reward_1  zD�9�,       ���E	ʞPA�z�A��7*

A2S/average_reward_1  zD炶�,       ���E	KZlC�z�A��7*

A2S/average_reward_1  zD+�,       ���E	�=_E�z�A�7*

A2S/average_reward_1  zD� �,       ���E	 gTG�z�Aѻ7*

A2S/average_reward_1  zD[���,       ���E	YI�z�A��7*

A2S/average_reward_1  zD0ѓ],       ���E	H�YK�z�A��7*

A2S/average_reward_1  zD�"�2,       ���E	6YM�z�A��7*

A2S/average_reward_1  zDa��I,       ���E	�TpO�z�A��7*

A2S/average_reward_1  zD�â�,       ���E	�vQ�z�A��7*

A2S/average_reward_1  zD��q,       ���E	�rS�z�A��7*

A2S/average_reward_1  zD�%xH,       ���E	A�|U�z�A��7*

A2S/average_reward_1  zD��,       ���E	��vW�z�A��7*

A2S/average_reward_1  zD~��s,       ���E	��]Y�z�A��8*

A2S/average_reward_1  zD4a�y,       ���E	J�\[�z�A�8*

A2S/average_reward_1  zD�B�,       ���E	)�Q]�z�Aɑ8*

A2S/average_reward_1  zDh�>�,       ���E	.�R_�z�A��8*

A2S/average_reward_1  zD��,       ���E	�Ja�z�A��8*

A2S/average_reward_1  zDi�r7,       ���E	�"@c�z�A��8*

A2S/average_reward_1  zD��O,       ���E	�5e�z�A�8*

A2S/average_reward_1  zD_8�},       ���E	�i@g�z�AѸ8*

A2S/average_reward_1  zD�GJ\,       ���E	wYi�z�A��8*

A2S/average_reward_1  zD�ꅭ,       ���E	�fJk�z�A��8*

A2S/average_reward_1  zD���,       ���E	��0m�z�A��8*

A2S/average_reward_1  zD�{�,       ���E	�>o�z�A��8*

A2S/average_reward_1  zD ~�,       ���E	�+�p�z�A��8*

A2S/average_reward_1  zD�8�,       ���E	���r�z�A��8*

A2S/average_reward_1  zD���,       ���E	ޮ�t�z�A��8*

A2S/average_reward_1  zD�FL1,       ���E	"�qv�z�A��8*

A2S/average_reward_1  zD5y,       ���E	�wgx�z�A��8*

A2S/average_reward_1  zD:���,       ���E	Ebz�z�A�9*

A2S/average_reward_1  zD��#�,       ���E	�b|�z�AɎ9*

A2S/average_reward_1  zD�Uh,       ���E	��b~�z�A��9*

A2S/average_reward_1  zD4�H,       ���E	fX��z�A��9*

A2S/average_reward_1  zD�s�,       ���E	�mg��z�A��9*

A2S/average_reward_1  zD:`��,       ���E	�{p��z�A�9*

A2S/average_reward_1  zD?HZB,       ���E	w���z�Aѵ9*

A2S/average_reward_1  zD����,       ���E	��|��z�A��9*

A2S/average_reward_1  zDk���,       ���E	�����z�A��9*

A2S/average_reward_1  zDB3J�,       ���E	Um��z�A��9*

A2S/average_reward_1  zD��H,       ���E	J�E��z�A��9*

A2S/average_reward_1  zDɁ��,       ���E	��8��z�A��9*

A2S/average_reward_1  zD���,       ���E	��.��z�A��9*

A2S/average_reward_1  zD�c�,       ���E	�s��z�A��9*

A2S/average_reward_1  zD��`�,       ���E	����z�A��9*

A2S/average_reward_1  zD$�+,       ���E	m���z�A��9*

A2S/average_reward_1  zD����,       ���E	$q��z�A�:*

A2S/average_reward_1  zD�>�1,       ���E	p����z�Aɋ:*

A2S/average_reward_1  zDe}u�,       ���E	����z�A��:*

A2S/average_reward_1  zD�i�,       ���E	����z�A��:*

A2S/average_reward_1  zD��,,       ���E	����z�A��:*

A2S/average_reward_1  zDQP��,       ���E	u�ף�z�A�:*

A2S/average_reward_1  zD����,       ���E	2V��z�AѬ:*

A2S/average_reward_1{xD'�>@,       ���E	_�-��z�A��:*

A2S/average_reward_1{xD6X,       ���E	��;��z�A��:*

A2S/average_reward_1{xDGo�,       ���E	%L<��z�A��:*

A2S/average_reward_1{xD�d��,       ���E	|.4��z�A��:*

A2S/average_reward_1{xD(O�&,       ���E	�z��z�A��:*

A2S/average_reward_1{xD��=H,       ���E	����z�A��:*

A2S/average_reward_1{xDs9�,       ���E	�~��z�A��:*

A2S/average_reward_1{xD�p`,       ���E	�!��z�A��:*

A2S/average_reward_1{xDh��,       ���E	N�=��z�A��:*

A2S/average_reward_1{xD% �,       ���E	�14��z�A��:*

A2S/average_reward_1{xDq���,       ���E	�W)��z�Aɂ;*

A2S/average_reward_1{xD2��,       ���E	�V���z�A��;*

A2S/average_reward_1{xDĳC�,       ���E	����z�A��;*

A2S/average_reward_1{xD-	�,       ���E	���z�A��;*

A2S/average_reward_1{xD�_�T,       ���E	{u���z�A�;*

A2S/average_reward_1{xD��,       ���E	����z�Aѩ;*

A2S/average_reward_1{xDA���,       ���E	���z�A�;*

A2S/average_reward_1��uD'7,       ���E	�����z�Aϱ;*

A2S/average_reward_1��uD�~� ,       ���E	�%���z�A��;*

A2S/average_reward_1��uD�5�,       ���E	q����z�A��;*

A2S/average_reward_1��uD���[,       ���E	����z�A��;*

A2S/average_reward_1��uD�g��,       ���E	t���z�A��;*

A2S/average_reward_1��uD�N(�,       ���E	�ߖ��z�A��;*

A2S/average_reward_1��uD7q$,       ���E	�����z�A��;*

A2S/average_reward_1��uDB{,       ���E	w���z�A��;*

A2S/average_reward_1��uD>�Y,       ���E	a!���z�A��;*

A2S/average_reward_1��uD�=<�,       ���E	ޑ���z�A��;*

A2S/average_reward_1��uDx��|,       ���E	42���z�A��;*

A2S/average_reward_1��uD��L(,       ���E	踧��z�AǇ<*

A2S/average_reward_1��uD���,       ���E	����z�A��<*

A2S/average_reward_1��uDi���,       ���E	���z�A��<*

A2S/average_reward_1��uD�H�,       ���E	�����z�A��<*

A2S/average_reward_1��uD0�.�,       ���E	0�v��z�A�<*

A2S/average_reward_1��uD�MZ,       ���E	�e��z�AϮ<*

A2S/average_reward_1��uD�y��,       ���E	�u_��z�A��<*

A2S/average_reward_1��uD��6,       ���E	�(U��z�A��<*

A2S/average_reward_1��uDˏ�,       ���E	�'1��z�A��<*

A2S/average_reward_1��uD��;,       ���E	G:��z�A��<*

A2S/average_reward_1��uD�3��,       ���E	wz���z�A��<*

A2S/average_reward_1��uD0��,       ���E	�w���z�A��<*

A2S/average_reward_1��uD�U�,       ���E	G���z�A��<*

A2S/average_reward_1��uD�ɧ,       ���E	�����z�A��<*

A2S/average_reward_1��uDZ��Z,       ���E	����z�A��<*

A2S/average_reward_1��uD|q�,       ���E	W�U��z�A��<*

A2S/average_reward_1��uDd�BB,       ���E	},A��z�AǄ=*

A2S/average_reward_1��uDY���,       ���E	�W0��z�A��=*

A2S/average_reward_1��uDV�"7,       ���E	�##��z�A��=*

A2S/average_reward_1��uD���,       ���E	�" �z�A��=*

A2S/average_reward_1��uD�d�,       ���E	��z�A�=*

A2S/average_reward_1��uD�d,       ���E	h9.�z�Aϫ=*

A2S/average_reward_1��uD�gY�,       ���E	jjJ�z�A��=*

A2S/average_reward_1��uD GNS,       ���E	D*X�z�A��=*

A2S/average_reward_1��uD��C�,       ���E	�D`
�z�A��=*

A2S/average_reward_1��uD%~7�,       ���E	��M�z�A��=*

A2S/average_reward_1��uD�6�,       ���E	a�E�z�A��=*

A2S/average_reward_1��uD�9��,       ���E	Y�P�z�A��=*

A2S/average_reward_1��uDZ)�1,       ���E	6�o�z�A��=*

A2S/average_reward_1��uDb���,       ���E	Qgb�z�A��=*

A2S/average_reward_1��uD	���,       ���E	�x�z�A��=*

A2S/average_reward_1��uD�_�	,       ���E	�x�z�A��=*

A2S/average_reward_1��uD� ��,       ���E	Ų��z�Aǁ>*

A2S/average_reward_1��uD�tp,       ���E	Q�j�z�A��>*

A2S/average_reward_1��uDi���,       ���E	(�t�z�A��>*

A2S/average_reward_1��uDʊ?,       ���E	�WU �z�A��>*

A2S/average_reward_1��uD�,��,       ���E	��H"�z�A�>*

A2S/average_reward_1��uD�g�;,       ���E	W�D$�z�AϨ>*

A2S/average_reward_1��uD��[,       ���E	��6&�z�A��>*

A2S/average_reward_1��uDav��,       ���E	0�(�z�A��>*

A2S/average_reward_1��uD��,       ���E	���)�z�A��>*

A2S/average_reward_1��uDw/��,       ���E	�s�+�z�A��>*

A2S/average_reward_1��uD�f�,       ���E	ߡ�-�z�A��>*

A2S/average_reward_1��uD)���,       ���E	��/�z�A��>*

A2S/average_reward_1��uD7O�,       ���E	��1�z�A��>*

A2S/average_reward_1��uD[�BV,       ���E	�ܝ3�z�A��>*

A2S/average_reward_1��uD�ƭ,       ���E	s��5�z�A��>*

A2S/average_reward_1��uD!Ӳm,       ���E	���7�z�A��>*

A2S/average_reward_1��uD��,       ���E	��s9�z�A��>*

A2S/average_reward_1��uD3y�N,       ���E	�3/;�z�A��?*

A2S/average_reward_1��uD�Ԓ=,       ���E	:�=�z�A��?*

A2S/average_reward_1��uD#v,       ���E	sG?�z�A��?*

A2S/average_reward_1��uD�x�:,       ���E	�A�z�A�?*

A2S/average_reward_1��uD�3$,       ���E	>�2C�z�Aϥ?*

A2S/average_reward_1��uD�1�,       ���E	qo)E�z�A��?*

A2S/average_reward_1��uD s�
,       ���E	&#G�z�A��?*

A2S/average_reward_1��uDw��=,       ���E	��I�z�A��?*

A2S/average_reward_1��uDb�,       ���E	��K�z�A��?*

A2S/average_reward_1��uDǨ|,       ���E	� 	M�z�A��?*

A2S/average_reward_1��uD����,       ���E	�!O�z�A��?*

A2S/average_reward_1��uD��       �v@�	gD�P�z�A��?*�

A2S/klSk?

A2S/average_advantage�.��

A2S/policy_network_lossGSQ�

A2S/value_network_loss���C

A2S/q_network_loss��C98�,       ���E	&[�P�z�A��?*

A2S/average_reward_1�.sD,       ���E	5	�P�z�A��?*

A2S/average_reward_1\�pD��J�,       ���E	 mQ�z�A��?*

A2S/average_reward_13�nDWZ>,       ���E	�Q�z�A��?*

A2S/average_reward_1�lDV���,       ���E	-�[Q�z�A��?*

A2S/average_reward_1R�iD '�U,       ���E	��Q�z�A��?*

A2S/average_reward_1��gD�O�Z,       ���E	-7�Q�z�A��?*

A2S/average_reward_1�)eDܙZO,       ���E	?ĨQ�z�A��?*

A2S/average_reward_1��bD����,       ���E	�S�Q�z�A��?*

A2S/average_reward_1�K`D���,       ���E	���Q�z�A��?*

A2S/average_reward_1f^D^Y0,       ���E	�F�Q�z�A��?*

A2S/average_reward_1��[D��y,       ���E	���Q�z�A��?*

A2S/average_reward_1�[D�y�,       ���E	�QR�z�A��?*

A2S/average_reward_1�XDۢ�K,       ���E	 �R�z�A��?*

A2S/average_reward_1�0VD8.`",       ���E	��ZR�z�A��?*

A2S/average_reward_1HTD�+K�,       ���E	(�R�z�A��?*

A2S/average_reward_1��QD[���,       ���E	�/�R�z�A��?*

A2S/average_reward_1=�ODd�,       ���E	O��R�z�A��?*

A2S/average_reward_1
'MD��D#,       ���E	�[�R�z�A��?*

A2S/average_reward_1q�JD���,       ���E	"g*S�z�A��?*

A2S/average_reward_1
wHD{nG,       ���E	�.8S�z�A��?*

A2S/average_reward_1�FD�N��,       ���E	��hS�z�A��?*

A2S/average_reward_1��CD�)��,       ���E		�uS�z�A��?*

A2S/average_reward_1�SAD���,,       ���E	ud�S�z�A��?*

A2S/average_reward_1
�>D I-�,       ���E	��S�z�A��?*

A2S/average_reward_1�<D��[�,       ���E	I� T�z�A��?*

A2S/average_reward_1��:D`C�&,       ���E	�T�z�A��?*

A2S/average_reward_1H8D@�c7,       ���E	}T�z�A��?*

A2S/average_reward_1�5D+��,       ���E	��PT�z�A��?*

A2S/average_reward_1��5D"8P,       ���E	ߣ�T�z�A��?*

A2S/average_reward_1�3D�,       ���E	�{�T�z�A��?*

A2S/average_reward_1��1D��c^,       ���E	�	+U�z�A��?*

A2S/average_reward_1�h/D� 1,       ���E	C6U�z�A��?*

A2S/average_reward_1R�,Ds���,       ���E	AU�z�A��?*

A2S/average_reward_1R�*D}��,       ���E	/�LU�z�A��?*

A2S/average_reward_1R(D���,       ���E	&��U�z�A��?*

A2S/average_reward_1��%D	���,       ���E	�6�U�z�A��?*

A2S/average_reward_1\�#D��5,       ���E	���U�z�A��?*

A2S/average_reward_1),!D动,       ���E	�PV�z�A��?*

A2S/average_reward_1��D�db�,       ���E	)V�z�A��?*

A2S/average_reward_1�Da�2,       ���E	 �LV�z�A��?*

A2S/average_reward_1{DD�-+g,       ���E	�j�V�z�A��?*

A2S/average_reward_1Dv`,       ���E	e�V�z�A��?*

A2S/average_reward_1͜D p.�,       ���E	hA�V�z�A��?*

A2S/average_reward_1�)D�u,       ���E	�V�z�A��?*

A2S/average_reward_1�D�&�,       ���E	!w�V�z�A��?*

A2S/average_reward_13CDͨ�,       ���E	f<�V�z�A��?*

A2S/average_reward_1��D�&!_,       ���E	q��V�z�A��?*

A2S/average_reward_1ף	D��^,       ���E	D$W�z�A��?*

A2S/average_reward_1�0D2�0�,       ���E	�5W�z�A��?*

A2S/average_reward_1 �D�N
.,       ���E	�nW�z�A��?*

A2S/average_reward_1�D ���,       ���E	�<yW�z�A��?*

A2S/average_reward_1�E Dh��J,       ���E	6��W�z�A��?*

A2S/average_reward_1ף�C�ojg,       ���E	�e�W�z�A��?*

A2S/average_reward_13��CqLܾ,       ���E	���W�z�A��?*

A2S/average_reward_1�u�C���/,       ���E	���W�z�A��?*

A2S/average_reward_1���C<�,       ���E	2�X�z�A��?*

A2S/average_reward_1��C@�Q,       ���E	�C*X�z�A��?*

A2S/average_reward_1 @�C���	,       ���E	fX�z�A��?*

A2S/average_reward_1���C�/==,       ���E	�YqX�z�A��?*

A2S/average_reward_1��CT�P�,       ���E	��xX�z�A��?*

A2S/average_reward_1.�C����,       ���E	��X�z�A��?*

A2S/average_reward_1H��C<���,       ���E	���X�z�A��?*

A2S/average_reward_1���CHC��,       ���E	] �X�z�A��?*

A2S/average_reward_1��C��#,       ���E	3��X�z�A��?*

A2S/average_reward_1\/�C�M[ ,       ���E	n��X�z�A��?*

A2S/average_reward_1�Q�CC?��,       ���E	d��X�z�A��?*

A2S/average_reward_1�q�C��q�,       ���E	�`�X�z�A��?*

A2S/average_reward_1=��C�4��,       ���E	-Y�z�A��?*

A2S/average_reward_1ף�C���,       ���E	wvY�z�A��?*

A2S/average_reward_1\��C�,       ���E	��Y�z�A��?*

A2S/average_reward_1�̦C0h3,       ���E	���Y�z�A��?*

A2S/average_reward_1=j�C���,       ���E	jc�Y�z�A��?*

A2S/average_reward_1��C��_Q,       ���E	}r	Z�z�A��?*

A2S/average_reward_1)�C}�3�,       ���E	V�@Z�z�A��?*

A2S/average_reward_1)��C"��',       ���E	P�{Z�z�A��?*

A2S/average_reward_1q�CX�w,       ���E	*��Z�z�A��?*

A2S/average_reward_1���CP���,       ���E	�u�Z�z�A��?*

A2S/average_reward_1{ԆC��`,       ���E	<$�Z�z�A��?*

A2S/average_reward_1��CO�",       ���E	.@�Z�z�A��?*

A2S/average_reward_1�zC�"�,       ���E	��[�z�A��?*

A2S/average_reward_1R�pC5�փ,       ���E	��![�z�A��?*

A2S/average_reward_1EgC_�ZT,       ���E	�a+[�z�A��?*

A2S/average_reward_1�z]C:3

,       ���E	��Y[�z�A��?*

A2S/average_reward_1 �TCaH��,       ���E	&G�[�z�A��?*

A2S/average_reward_1RxKCc�x,       ���E	[�z�A��?*

A2S/average_reward_1R�AC�s�0,       ���E	�Y�[�z�A��?*

A2S/average_reward_1H�7C�8,       ���E	gE�[�z�A��?*

A2S/average_reward_1\.CǷ��,       ���E	%��[�z�A��?*

A2S/average_reward_1�z$CE�*u,       ���E	���[�z�A��?*

A2S/average_reward_1ףC���,       ���E	��[�z�A��@*

A2S/average_reward_1õC�?�,       ���E	m�
\�z�A��@*

A2S/average_reward_1�C�^l,       ���E	��M\�z�A��@*

A2S/average_reward_1�z�B�v��,       ���E	�3_\�z�A��@*

A2S/average_reward_1{�BX!�,       ���E	'j\�z�A́@*

A2S/average_reward_1�z�B�m,       ���E	z��\�z�A��@*

A2S/average_reward_1���B���,       ���E	�K�\�z�A��@*

A2S/average_reward_1\�BgVW,       ���E	�z@]�z�A��@*

A2S/average_reward_1��BV�x4,       ���E	kL]�z�A��@*

A2S/average_reward_1�u�B�T�,       ���E	:[]�z�A��@*

A2S/average_reward_1{zBw�,       ���E	�X�]�z�A��@*

A2S/average_reward_1=
~B�5�%,       ���E	Y��]�z�A��@*

A2S/average_reward_1���B�$R�,       ���E	1�]�z�A��@*

A2S/average_reward_1�~B2ѥ�,       ���E	D�^�z�A��@*

A2S/average_reward_1{��B/],       ���E	��H^�z�A�@*

A2S/average_reward_1ff�B2rQ%,       ���E	t�^�z�A�@*

A2S/average_reward_1��B򍻮,       ���E	Yk�^�z�A��@*

A2S/average_reward_1=��By���,       ���E	9^�z�A��@*

A2S/average_reward_1�p�By!�,       ���E	�: _�z�Aϋ@*

A2S/average_reward_1
׃B/�F,       ���E	V_�z�A�@*

A2S/average_reward_1��B�*lg,       ���E	r�@_�z�A@*

A2S/average_reward_1��B3�Ћ,       ���E	��O_�z�A܌@*

A2S/average_reward_1)܃BՏƜ,       ���E	�Z_�z�A�@*

A2S/average_reward_1�уB�T;�,       ���E	��g_�z�A��@*

A2S/average_reward_1��B���,       ���E	Bo�_�z�Aލ@*

A2S/average_reward_1���BUC�3,       ���E	��_�z�A��@*

A2S/average_reward_1 ��B3&�2,       ���E	���_�z�Aێ@*

A2S/average_reward_1���B���,       ���E	��`�z�A��@*

A2S/average_reward_133�B�t�,       ���E	<C`�z�A��@*

A2S/average_reward_1 ��B�[M�,       ���E	׎P`�z�A��@*

A2S/average_reward_1R8�Bs�,       ���E	�```�z�AՐ@*

A2S/average_reward_1�L�B3�>i,       ���E	��n`�z�A�@*

A2S/average_reward_1f�B1Q&M,       ���E	�}}`�z�A��@*

A2S/average_reward_1
׀B?Q_w,       ���E	27�`�z�A��@*

A2S/average_reward_1�рB1S��,       ���E	L�`�z�A��@*

A2S/average_reward_1{}B6��w,       ���E	�ڤ`�z�Aґ@*

A2S/average_reward_1��yB;�d�,       ���E	�y�`�z�Aْ@*

A2S/average_reward_1�G~BDK,       ���E	D�`�z�A��@*

A2S/average_reward_1\�~Bp�l,       ���E	IZa�z�Aʔ@*

A2S/average_reward_1R8�B���,       ���E	hs�a�z�A��@*

A2S/average_reward_1�B��!W,       ���E	��a�z�A��@*

A2S/average_reward_1��yB��iQ,       ���E	�e�a�z�A��@*

A2S/average_reward_133{BZ9�,       ���E	���a�z�AԖ@*

A2S/average_reward_1q={B���,       ���E	#
�a�z�A�@*

A2S/average_reward_1�({B�
��,       ���E	��b�z�Aח@*

A2S/average_reward_1�z~B�Ա�,       ���E	A�Kb�z�A��@*

A2S/average_reward_1ף~Bvb�,       ���E	u�b�z�A��@*

A2S/average_reward_1�G�B���{,       ���E	v5c�z�A��@*

A2S/average_reward_1�уB/�S,       ���E	��4c�z�A��@*

A2S/average_reward_1ף�B����,       ���E	r�c�z�A��@*

A2S/average_reward_1  �B�J��,       ���E	C�c�z�A��@*

A2S/average_reward_1�(�B��U�,       ���E	/��c�z�A�@*

A2S/average_reward_1H�B���x,       ���E	���c�z�A��@*

A2S/average_reward_1���BX)�Y,       ���E	�6d�z�A��@*

A2S/average_reward_1��B�:��,       ���E	�d�z�A��@*

A2S/average_reward_1�Q�B�8L�,       ���E	'�}d�z�A��@*

A2S/average_reward_1f�ByaS-,       ���E	�s�d�z�A��@*

A2S/average_reward_1{��B`-��,       ���E	F�d�z�A��@*

A2S/average_reward_1)܇BM>�,       ���E	�M�d�z�A֡@*

A2S/average_reward_1\�B��B,       ���E	���d�z�A͢@*

A2S/average_reward_1�u�B�9k       �v@�	;"%e�z�A͢@*�

A2S/klo�?

A2S/average_advantage:|�=

A2S/policy_network_loss7_�?

A2S/value_network_loss:m�C

A2S/q_network_loss#L�C2��,       ���E	���e�z�A�@*

A2S/average_reward_1��B��.,       ���E	d��f�z�A��@*

A2S/average_reward_1R��BA���,       ���E	�m�g�z�A߬@*

A2S/average_reward_1)\�B뭯),       ���E	���h�z�A��@*

A2S/average_reward_1R��B�姲,       ���E	�r�i�z�Aϵ@*

A2S/average_reward_1�̲BJ�,       ���E	��sj�z�A��@*

A2S/average_reward_1�u�B8%��,       ���E	�Tuk�z�Aѻ@*

A2S/average_reward_1\��B�(D�,       ���E	��l�z�A��@*

A2S/average_reward_1�(�B�;,       ���E	qj�m�z�A��@*

A2S/average_reward_1���B��#,       ���E	���n�z�A��@*

A2S/average_reward_1)\�B�6�b,       ���E	ܿo�z�A��@*

A2S/average_reward_1�Q�Bf<�%,       ���E	~�p�z�A��@*

A2S/average_reward_1���B�'�A,       ���E	�Ȇq�z�A��@*

A2S/average_reward_1���BRj1�,       ���E	
�jr�z�A��@*

A2S/average_reward_1=
�B[�mb,       ���E	r��r�z�A��@*

A2S/average_reward_1R�Ci�$,       ���E	,U�s�z�A��@*

A2S/average_reward_13�Cɶ[,       ���E	�u�z�A��@*

A2S/average_reward_1nC(��I,       ���E	#5�v�z�A��@*

A2S/average_reward_1)\C�ڵk,       ���E	��w�z�A��@*

A2S/average_reward_1{C9D��,       ���E	�'�x�z�A��@*

A2S/average_reward_1�C)�&*,       ���E	�V�y�z�A��@*

A2S/average_reward_1=� C���Q,       ���E	�f�z�z�A��@*

A2S/average_reward_1
�%C���b,       ���E	�M�{�z�A��@*

A2S/average_reward_1�k)C+b"+,       ���E	�o|�z�A��@*

A2S/average_reward_1f�+C![,       ���E	��|�z�A��A*

A2S/average_reward_1�z-Cb#� ,       ���E	Rx�}�z�A�A*

A2S/average_reward_1R80C��d�,       ���E	e�~�z�A��A*

A2S/average_reward_1�3C�$�),       ���E	/0�z�A։A*

A2S/average_reward_1 @6C�a-,       ���E	JW��z�A��A*

A2S/average_reward_1�9C�8m,       ���E	�s؀�z�A��A*

A2S/average_reward_1��=C`�f�,       ���E	YIG��z�AܑA*

A2S/average_reward_1q=?C-ؾZ,       ���E	bYk��z�A��A*

A2S/average_reward_1��DC�!�,       ���E	���z�A��A*

A2S/average_reward_1�LGC7X��,       ���E	.E��z�A��A*

A2S/average_reward_1)\LC�k)�,       ���E	SV��z�A��A*

A2S/average_reward_1f�QC�P�,       ���E	�q���z�A��A*

A2S/average_reward_1RxTC8!�,       ���E	\����z�A�A*

A2S/average_reward_1�GWC�-��,       ���E	��9��z�A��A*

A2S/average_reward_1�cZC3�F�,       ���E	�
��z�A�A*

A2S/average_reward_1{T^C;#�,       ���E	����z�A��A*

A2S/average_reward_1=�cCu�#�,       ���E	_�-��z�A��A*

A2S/average_reward_1{�gC�x�,       ���E	c�C��z�AݹA*

A2S/average_reward_1�YmC���,       ���E	ak��z�A��A*

A2S/average_reward_1��pC��R�,       ���E	뫌�z�AɿA*

A2S/average_reward_1)\sC�ޫ+,       ���E	��S��z�A��A*

A2S/average_reward_1�cvC�}v�,       ���E	��;��z�A��A*

A2S/average_reward_1�BzCID,       ���E	J�"��z�A��A*

A2S/average_reward_1\�}CQ;e,       ���E	�@A��z�A��A*

A2S/average_reward_1͌�C@�u�,       ���E	@����z�A��A*

A2S/average_reward_1)��C�Whc,       ���E	Ճ/��z�A��A*

A2S/average_reward_1
W�Ch��h,       ���E	R�5��z�A��A*

A2S/average_reward_1H��C���,       ���E	Rs���z�A��A*

A2S/average_reward_1�,�C�4�,       ���E	eݓ�z�A��A*

A2S/average_reward_1׃�C?�z?,       ���E	�����z�A��A*

A2S/average_reward_1=ʊC-���,       ���E	Z+���z�A��A*

A2S/average_reward_1)��C�xF,       ���E	 J��z�A��A*

A2S/average_reward_1)\�CB��,       ���E	W���z�A��A*

A2S/average_reward_1f�C�i��,       ���E	����z�A��A*

A2S/average_reward_1)<�C�	�,       ���E	�Տ��z�A��A*

A2S/average_reward_1�ÓC��#,       ���E	�C<��z�A��A*

A2S/average_reward_1�՗C�T��,       ���E	6�+��z�A��A*

A2S/average_reward_1ΙC���,       ���E	m&��z�A�B*

A2S/average_reward_1�C�Cg��,       ���E	c���z�A��B*

A2S/average_reward_1�y�C�s,       ���E	Z��z�AلB*

A2S/average_reward_1 ��CX'��,       ���E	B��z�AĈB*

A2S/average_reward_1���C��P�,       ���E	����z�A��B*

A2S/average_reward_1�9�C
�ך,       ���E	}���z�A��B*

A2S/average_reward_1׃�C,$�,       ���E	1q���z�A��B*

A2S/average_reward_1͌�C��rB,       ���E	5����z�A��B*

A2S/average_reward_1���C��,       ���E	ɑ��z�A��B*

A2S/average_reward_1  �C��,       ���E	j};��z�A�B*

A2S/average_reward_1ᚫCF�#�,       ���E	��e��z�AơB*

A2S/average_reward_1��C�B�,       ���E	%7p��z�AҥB*

A2S/average_reward_1�C2���,       ���E	=�X��z�AةB*

A2S/average_reward_1
w�C숾,       ���E	mel��z�A�B*

A2S/average_reward_1f�C���',       ���E	�ah��z�AֱB*

A2S/average_reward_1
W�C!�D,       ���E	�s3��z�A��B*

A2S/average_reward_1�ȹC��(,       ���E	����z�A��B*

A2S/average_reward_1
׻C��!:,       ���E	A�ګ�z�AջB*

A2S/average_reward_1�ڼC����,       ���E	:�ˬ�z�AſB*

A2S/average_reward_1�޾CvF�V,       ���E	�����z�A��B*

A2S/average_reward_1=
�CmhN,       ���E	�K���z�A��B*

A2S/average_reward_1
��C��$�,       ���E	�oD��z�A��B*

A2S/average_reward_1�K�C2�@V,       ���E	����z�A��B*

A2S/average_reward_1���C��A0,       ���E	�̈��z�A��B*

A2S/average_reward_1
��C����,       ���E	��"��z�A��B*

A2S/average_reward_1H��C�f�,       ���E	��-��z�A��B*

A2S/average_reward_1�g�C�S�n,       ���E	sͅ��z�A��B*

A2S/average_reward_1���C '/,       ���E	Ӄ���z�A��B*

A2S/average_reward_1�#�C��,       ���E	�ĵ�z�A��B*

A2S/average_reward_1�C�Ϫ ,       ���E	A3���z�A��B*

A2S/average_reward_1��C�0�,       ���E	t�;��z�A��B*

A2S/average_reward_1{T�C�iK,       ���E	��5��z�A��B*

A2S/average_reward_1��Cy9��,       ���E	�i2��z�A��B*

A2S/average_reward_1�5�C�\�`,       ���E	I�G��z�A��B*

A2S/average_reward_1��C����,       ���E	��ݼ�z�A��B*

A2S/average_reward_1)�C+@�,       ���E	l�ݽ�z�A��C*

A2S/average_reward_1�q�C=J,       ���E	K�;�z�A��C*

A2S/average_reward_1=��Ca=��,       ���E	 ���z�A��C*

A2S/average_reward_1)�C�x��,       ���E	����z�A��C*

A2S/average_reward_1=��C�J�,       ���E	����z�A�C*

A2S/average_reward_1R8�C�Z�,       ���E	y����z�A��C*

A2S/average_reward_1��C%�,       ���E	�3t��z�A��C*

A2S/average_reward_1=j�C��T,       ���E	��>��z�AҜC*

A2S/average_reward_1��C$ǔ,       ���E	cE���z�A��C*

A2S/average_reward_1�Y�C���*,       ���E	����z�A��C*

A2S/average_reward_1H!�C�Y��,       ���E	�՚��z�A��C*

A2S/average_reward_1
w�C@�M,       ���E	h���z�A��C*

A2S/average_reward_1���C�~��,       ���E	�����z�A�C*

A2S/average_reward_1=J�CWHf�,       ���E	}ֶ��z�A��C*

A2S/average_reward_1=��CΗ~W,       ���E	�����z�A��C*

A2S/average_reward_1���C��u$,       ���E	�'���z�A��C*

A2S/average_reward_13S�C��c@,       ���E	���z�A��C*

A2S/average_reward_1�~�Cw{�',       ���E	����z�A��C*

A2S/average_reward_1
��CX@l�,       ���E	�x��z�A��C*

A2S/average_reward_1��C.I�S,       ���E	��
��z�A��C*

A2S/average_reward_1�Q�CUc�m,       ���E	0����z�A��C*

A2S/average_reward_1���Cv�ʗ,       ���E	���z�A��C*

A2S/average_reward_1��C��,       ���E	k-|��z�A��C*

A2S/average_reward_1���C���T,       ���E	N��z�A��C*

A2S/average_reward_1\��C�"�],       ���E	W*Z��z�A��C*

A2S/average_reward_1��C-0^�,       ���E	��_��z�A��C*

A2S/average_reward_1{��C�t��,       ���E	����z�A��C*

A2S/average_reward_1=
�C�j&�,       ���E	{]��z�A��C*

A2S/average_reward_1RX�C&=a�,       ���E	��	��z�A��C*

A2S/average_reward_1{��C?De�,       ���E	׵��z�A��C*

A2S/average_reward_133�C��k�,       ���E	h���z�A��C*

A2S/average_reward_1ff�C�ћ�,       ���E	�܈��z�A��C*

A2S/average_reward_1=J�C66�t,       ���E	o#��z�A��C*

A2S/average_reward_1�:�C~ٌ�,       ���E	!����z�A��C*

A2S/average_reward_1���CSs�,       ���E	)���z�A��C*

A2S/average_reward_1��CB�,       ���E	�����z�A��D*

A2S/average_reward_1���C��'�,       ���E	Rk��z�A��D*

A2S/average_reward_1R��C�n�>,       ���E	�p���z�A��D*

A2S/average_reward_1.�C؆��,       ���E	�/��z�A��D*

A2S/average_reward_1
7�C+T/,       ���E	���z�A�D*

A2S/average_reward_1���C9>��,       ���E	�S���z�AْD*

A2S/average_reward_1���C�\7,       ���E	x���z�A��D*

A2S/average_reward_1{��C��W,       ���E	�C���z�AؘD*

A2S/average_reward_1���C��C,       ���E	�N/��z�A��D*

A2S/average_reward_1\��C��/,       ���E		"��z�A��D*

A2S/average_reward_1  �ChS�,       ���E	4m���z�A��D*

A2S/average_reward_1H!�Ca�&�,       ���E	�Y���z�AǥD*

A2S/average_reward_1\��C�m�,       ���E	ֈ��z�A��D*

A2S/average_reward_1=��C�8,       ���E	�u��z�A��D*

A2S/average_reward_1N�C��2,       ���E	2�|��z�A�D*

A2S/average_reward_1��CT�0�,       ���E	x���z�A��D*

A2S/average_reward_1��C���B,       ���E	�tu��z�A�D*

A2S/average_reward_1  �CN{�8,       ���E	s2���z�A��D*

A2S/average_reward_1
w�C�s�,       ���E	�����z�A��D*

A2S/average_reward_1{t�C#ֆ�,       ���E	Z2-��z�AϾD*

A2S/average_reward_1��CB{�~,       ���E	c���z�A��D*

A2S/average_reward_1���C ��,       ���E	f����z�A��D*

A2S/average_reward_1H��CEd��,       ���E	�w��z�A��D*

A2S/average_reward_1HA�C��,       ���E	X���z�A��D*

A2S/average_reward_1�~�C�,d,       ���E	Z���z�A��D*

A2S/average_reward_1���Cϣ_,       ���E	h���z�A��D*

A2S/average_reward_1���Cp�,       ���E	ψ���z�A��D*

A2S/average_reward_1R8�Ch`��,       ���E	�����z�A��D*

A2S/average_reward_1��Cu��,       ���E	tX��z�A��D*

A2S/average_reward_1)��Cב���       �v@�	�D(��z�A��D*�

A2S/klXP1?

A2S/average_advantage�-7>

A2S/policy_network_loss��>

A2S/value_network_loss9��B

A2S/q_network_loss���B�<��,       ���E	�B���z�A��D*

A2S/average_reward_1{T�CR���,       ���E	�zH��z�A��D*

A2S/average_reward_13S�C3��,       ���E	����z�A��D*

A2S/average_reward_1���C;�fh,       ���E	�W���z�A��D*

A2S/average_reward_1RX�C S�,       ���E	��|��z�A��D*

A2S/average_reward_1�+�C��d,       ���E	Ut��z�A��D*

A2S/average_reward_1���C.T�,       ���E	�P(��z�A��E*

A2S/average_reward_1E�C��ii,       ���E	%� �z�A��E*

A2S/average_reward_1��C�y,       ���E	Y�� �z�A��E*

A2S/average_reward_1�C�C���,       ���E	}[��z�A��E*

A2S/average_reward_1{T�C`h�,       ���E	c���z�A��E*

A2S/average_reward_1q��C�C�D,       ���E	�M�z�A��E*

A2S/average_reward_1)��CA��O,       ���E	���z�AڛE*

A2S/average_reward_1ff�C��0,       ���E	.U��z�A��E*

A2S/average_reward_1���C9��8,       ���E	P���z�AȨE*

A2S/average_reward_1e�C9*�d,       ���E	J�	�z�A��E*

A2S/average_reward_1�q�C"g��,       ���E	`(
�z�A�E*

A2S/average_reward_1�"�C���W,       ���E	���z�AɵE*

A2S/average_reward_1���Cb��,       ���E	O��z�A��E*

A2S/average_reward_1{��C���,       ���E	)���z�A��E*

A2S/average_reward_1�y�C.���,       ���E	I���z�A��E*

A2S/average_reward_1R8�C\я�,       ���E	���z�A��E*

A2S/average_reward_1�q�CX�°,       ���E	V��z�A��E*

A2S/average_reward_1�q�C'�&�,       ���E	��d�z�A��E*

A2S/average_reward_1
7�C�q�,       ���E	��%�z�A��E*

A2S/average_reward_1���C��>;,       ���E	���z�A��E*

A2S/average_reward_1R�Dd7�,       ���E	��$�z�A��E*

A2S/average_reward_1{�D��,       ���E	���z�A��E*

A2S/average_reward_1\D��,       ���E	����z�A��E*

A2S/average_reward_1{4D��p�,       ���E	���z�A��E*

A2S/average_reward_1f�D��]�,       ���E	�ln�z�A��E*

A2S/average_reward_1
WD4lt,       ���E	� �z�AلF*

A2S/average_reward_1�D4M\�,       ���E	��~!�z�A��F*

A2S/average_reward_1{4D&3�,       ���E	IB#�z�AސF*

A2S/average_reward_1HD~�d,       ���E	��"$�z�A��F*

A2S/average_reward_1{�D�Dδ,       ���E	�+>%�z�A�F*

A2S/average_reward_1DX�8�,       ���E	�D'�z�A֠F*

A2S/average_reward_1)\D��s,       ���E	p	�(�z�A��F*

A2S/average_reward_1 0D�F�,       ���E	��}*�z�A��F*

A2S/average_reward_1�sD�4�\,       ���E	��m+�z�A��F*

A2S/average_reward_1)D�Y��,       ���E	�wd-�z�A��F*

A2S/average_reward_1�D]K<,       ���E	���-�z�A��F*

A2S/average_reward_1�nD���J,       ���E	G�/�z�A��F*

A2S/average_reward_1{�D�
�,       ���E	|j1�z�A��F*

A2S/average_reward_1\	D����,       ���E	�V�2�z�A��F*

A2S/average_reward_1��	D(k˫,       ���E	2w4�z�A��F*

A2S/average_reward_1�
D���k,       ���E	�;n5�z�A��F*

A2S/average_reward_1� 
D=�^,       ���E	mW7�z�A��F*

A2S/average_reward_1q�D;��,       ���E	�O9�z�A��F*

A2S/average_reward_1{$D|�,       ���E	�*:�z�A��F*

A2S/average_reward_1�XD��A�,       ���E	���;�z�A��F*

A2S/average_reward_1��D�)�,       ���E	p+=�z�A��F*

A2S/average_reward_13sD==��,       ���E	�t|>�z�A��F*

A2S/average_reward_1��Dۍ��,       ���E	-XV?�z�A��G*

A2S/average_reward_1��Dԕt�,       ���E	]��@�z�A��G*

A2S/average_reward_1�D��,       ���E	gJyA�z�AǈG*

A2S/average_reward_1��Dc�d�,       ���E	��nB�z�A�G*

A2S/average_reward_1�
D׭�Q,       ���E	��QC�z�A��G*

A2S/average_reward_1��D1o�,       ���E	���C�z�A��G*

A2S/average_reward_1
D$w�,       ���E	��OE�z�AǗG*

A2S/average_reward_1=jD]�Lq,       ���E	VתE�z�A��G*

A2S/average_reward_1 PDs ,       ���E	�̈F�z�A��G*

A2S/average_reward_1{$D`���,       ���E	�� G�z�A�G*

A2S/average_reward_1q-Dl���,       ���E	�
I�z�A٦G*

A2S/average_reward_1�RD�a��,       ���E	��J�z�A��G*

A2S/average_reward_1\�DD��X,       ���E	�K�z�A�G*

A2S/average_reward_1{dDg�q�,       ���E	&�M�z�AٹG*

A2S/average_reward_1�#D�q��,       ���E	�	O�z�A��G*

A2S/average_reward_1׳D�rޙ,       ���E	Bj�P�z�A��G*

A2S/average_reward_1�HDI���,       ���E	�ZQ�z�A��G*

A2S/average_reward_1H�D�Jέ,       ���E	�Q�Q�z�A��G*

A2S/average_reward_1�PD�-e�,       ���E	B�S�z�A��G*

A2S/average_reward_1ÅD�\H�,       ���E	��T�z�A��G*

A2S/average_reward_1HQD��(!,       ���E	�SU�z�A��G*

A2S/average_reward_1q�D#��,       ���E	�=�U�z�A��G*

A2S/average_reward_1)D��`,       ���E	���V�z�A��G*

A2S/average_reward_1{TD���,       ���E	��W�z�A��G*

A2S/average_reward_1�D��FP,       ���E	~�Y�z�A��G*

A2S/average_reward_1�rD�DH�,       ���E	�RpZ�z�A��G*

A2S/average_reward_1q�D%���,       ���E	�\�z�A��G*

A2S/average_reward_1׃D�b�*,       ���E	�.^�z�A��G*

A2S/average_reward_1{D�I5,       ���E	��_�z�A��G*

A2S/average_reward_1�*D;��,       ���E	BB�_�z�A��H*

A2S/average_reward_1q�D���,       ���E	C�_`�z�A��H*

A2S/average_reward_1)D�h�,       ���E	R�a�z�A��H*

A2S/average_reward_1NDF�T,       ���E	4��c�z�A�H*

A2S/average_reward_1�3Do�ݲ,       ���E	��d�z�A��H*

A2S/average_reward_1��D#��,       ���E	��d�z�A��H*

A2S/average_reward_1)LD!�i�,       ���E	hue�z�A�H*

A2S/average_reward_1�rDg�~f,       ���E	��mg�z�AӟH*

A2S/average_reward_1׳DE|�,       ���E	�wh�z�A��H*

A2S/average_reward_1 PDtx�,       ���E	��j�z�AǪH*

A2S/average_reward_1��D8�,       ���E	Ȇ�j�z�A�H*

A2S/average_reward_1��D�PU!,       ���E	�M�k�z�A��H*

A2S/average_reward_1H�D�Z	�,       ���E	k��l�z�AеH*

A2S/average_reward_13sD� s,       ���E	O+�m�z�AȹH*

A2S/average_reward_1fVD�L��,       ���E	�oo�z�AнH*

A2S/average_reward_1�YD���,       ���E	��o�z�A��H*

A2S/average_reward_1�wDύi,       ���E	�hgq�z�A��H*

A2S/average_reward_1�|DB��,       ���E	��Kr�z�A��H*

A2S/average_reward_1�LD��ԣ,       ���E	ow_r�z�A��H*

A2S/average_reward_13�D~d�!,       ���E	Ъs�z�A��H*

A2S/average_reward_1�ND�e`z,       ���E	�t�t�z�A��H*

A2S/average_reward_1\oD�=,       ���E	��rv�z�A��H*

A2S/average_reward_1f�D�^s�,       ���E	��Gw�z�A��H*

A2S/average_reward_1)�D�@
�,       ���E	�Ry�z�A��H*

A2S/average_reward_1)�DL���,       ���E	R�z�z�A��H*

A2S/average_reward_1q�D��tG,       ���E	�c|�z�A��H*

A2S/average_reward_1fFD��Y�,       ���E	{�~�z�A��H*

A2S/average_reward_1��D�!�,       ���E	�~�z�A��H*

A2S/average_reward_1HQD��)�,       ���E	�K��z�A��H*

A2S/average_reward_1�nD�z
),       ���E	V����z�A��I*

A2S/average_reward_1=�DqM�,       ���E	�?Â�z�AËI*

A2S/average_reward_1��D"�2,       ���E	��e��z�A��I*

A2S/average_reward_1�D%¸�,       ���E	�Y"��z�A�I*

A2S/average_reward_1 @D�t=�,       ���E	w\f��z�AʕI*

A2S/average_reward_1�D�m,       ���E	����z�A��I*

A2S/average_reward_1�^D��wA,       ���E	:#v��z�A��I*

A2S/average_reward_1�^D9a�P,       ���E	���z�A��I*

A2S/average_reward_1��D,       ���E		��z�AƠI*

A2S/average_reward_1��Dl&�p,       ���E	���z�A�I*

A2S/average_reward_1�#D� �,       ���E	D����z�A��I*

A2S/average_reward_1� DB��,       ���E	v��z�AɧI*

A2S/average_reward_1�.D��ޮ,       ���E	#�Ê�z�A��I*

A2S/average_reward_1��D����,       ���E	x�,��z�A��I*

A2S/average_reward_1)<DNp6,       ���E	ڰ��z�AӶI*

A2S/average_reward_1 �D��,       ���E	7Rގ�z�A��I*

A2S/average_reward_1��D�[),       ���E	�����z�A��I*

A2S/average_reward_1��Dp�^,       ���E	͵���z�A��I*

A2S/average_reward_1��D����,       ���E	B�Ǒ�z�A��I*

A2S/average_reward_1�Db;�},       ���E	�����z�A��I*

A2S/average_reward_13�D)a�b,       ���E	�����z�A��I*

A2S/average_reward_1R�DJa�,       ���E	ᣟ��z�A��I*

A2S/average_reward_1{4Dv��,       ���E	�����z�A��I*

A2S/average_reward_1��D��^,       ���E	�)3��z�A��I*

A2S/average_reward_1�GD��՗,       ���E	�ٽ��z�A��I*

A2S/average_reward_1
�D�!�m,       ���E	.S���z�A��I*

A2S/average_reward_1R(D�h,       ���E	an!��z�A��I*

A2S/average_reward_1�D2���,       ���E	����z�A��I*

A2S/average_reward_1{D�XV%,       ���E	��؞�z�A��I*

A2S/average_reward_1f6D����,       ���E	�PJ��z�A��I*

A2S/average_reward_1�HD�:M,       ���E	'�*��z�A��J*

A2S/average_reward_1fD�IKD,       ���E	o&��z�AȇJ*

A2S/average_reward_1=ZD�3tI,       ���E	0�?��z�A��J*

A2S/average_reward_1�D��),       ���E	� ���z�A��J*

A2S/average_reward_1{TDh���,       ���E	��¤�z�A�J*

A2S/average_reward_1�BD�b(�,       ���E	3��z�A֖J*

A2S/average_reward_1��D1��,       ���E	�9Ƨ�z�A��J*

A2S/average_reward_1ffD��m�,       ���E	e~*��z�A��J*

A2S/average_reward_1��D~^9,       ���E	�i���z�AߨJ*

A2S/average_reward_1�JD�7r,       ���E	��$��z�A�J*

A2S/average_reward_1HD��U�,       ���E	q���z�AȲJ*

A2S/average_reward_1q�DeQ��,       ���E	�q���z�A��J*

A2S/average_reward_1�*D؎��,       ���E	3�>��z�A��J*

A2S/average_reward_1=�D�y��,       ���E	dN��z�A��J*

A2S/average_reward_1 �D�Q��,       ���E	�����z�A��J*

A2S/average_reward_1f�D�~��,       ���E	:�Ǳ�z�A��J*

A2S/average_reward_1� D���,       ���E	����z�A��J*

A2S/average_reward_1�<D�i�O,       ���E	�Y���z�A��J*

A2S/average_reward_1��D�*�,       ���E	Su���z�A��J*

A2S/average_reward_1��D�SN�       �v@�	��ǵ�z�A��J*�

A2S/kl |R?

A2S/average_advantageΦD�

A2S/policy_network_loss���

A2S/value_network_loss]5}C

A2S/q_network_loss�xC��"�,       ���E	���z�A��J*

A2S/average_reward_1��D$hX,       ���E	Ok��z�A��J*

A2S/average_reward_1
�D���,       ���E	_@��z�A��J*

A2S/average_reward_1{�D	5�2,       ���E	�����z�A��J*

A2S/average_reward_1=ZD@�C�,       ���E	�5޺�z�A��J*

A2S/average_reward_133D���7,       ���E	Z����z�A��J*

A2S/average_reward_1�Df��,       ���E	@d��z�A��J*

A2S/average_reward_1�	
D*,       ���E	����z�A��J*

A2S/average_reward_1�'	D�-}�,       ���E	 �k��z�A��J*

A2S/average_reward_1\�D7�9�,       ���E	�u̽�z�A��J*

A2S/average_reward_1f�Ddv<,       ���E	Hdÿ�z�A��J*

A2S/average_reward_1Å	D�I��,       ���E	�u��z�A��J*

A2S/average_reward_1��D)�a,       ���E	p��z�A��J*

A2S/average_reward_1R8D����,       ���E	�>��z�A��J*

A2S/average_reward_1R(Dm{�,       ���E	��P��z�A��K*

A2S/average_reward_1\�DRb},       ���E	"���z�A��K*

A2S/average_reward_1=�DV�,       ���E	�`@��z�AˊK*

A2S/average_reward_1D����,       ���E	�Vi��z�A��K*

A2S/average_reward_1�D��ό,       ���E	�5���z�A��K*

A2S/average_reward_1{�Dl,;�,       ���E	�����z�A��K*

A2S/average_reward_1R�Df���,       ���E	� ���z�A��K*

A2S/average_reward_1��DH���,       ���E	���z�A��K*

A2S/average_reward_1Hq D�h�,       ���E	����z�A��K*

A2S/average_reward_1�p D�a0�,       ���E	 �V��z�A��K*

A2S/average_reward_1=j�C�>q,       ���E	,4���z�A�K*

A2S/average_reward_1���C4ґr,       ���E	��U��z�A��K*

A2S/average_reward_1�P�C�>��,       ���E	ꤕ��z�A��K*

A2S/average_reward_1�"�C	>-,       ���E	����z�A��K*

A2S/average_reward_1\/�CP�֒,       ���E	�<��z�A��K*

A2S/average_reward_1���C��qu,       ���E	�����z�A��K*

A2S/average_reward_1�c�C���,       ���E	����z�A��K*

A2S/average_reward_1=��C!$5,       ���E	����z�A��K*

A2S/average_reward_1���C����,       ���E	]\���z�A��K*

A2S/average_reward_1�b�C)�,       ���E	�:���z�AݨK*

A2S/average_reward_1�Q�CrV�,       ���E	�g��z�A��K*

A2S/average_reward_1f��C�D�#,       ���E	��T��z�A��K*

A2S/average_reward_1��C̫�q,       ���E	�A���z�AįK*

A2S/average_reward_1q��C��}[,       ���E	x���z�A��K*

A2S/average_reward_1��C��8,       ���E	�4��z�A۵K*

A2S/average_reward_1ͬ�Cz���,       ���E	�]��z�A��K*

A2S/average_reward_1���C<2,       ���E	�_��z�A�K*

A2S/average_reward_1��C�^,       ���E	�,��z�A��K*

A2S/average_reward_1=*�C=y!,       ���E	sl���z�A��K*

A2S/average_reward_1H��C�<},       ���E	vg���z�A��K*

A2S/average_reward_1fF�C&�>�,       ���E	�q��z�A��K*

A2S/average_reward_1H��CcbGe,       ���E	�e���z�A��K*

A2S/average_reward_1)\�CG�λ,       ���E	�c���z�A��K*

A2S/average_reward_1��C��y,       ���E	F]���z�A��K*

A2S/average_reward_1���C�n?,       ���E	Н��z�A��K*

A2S/average_reward_1�#�C���,       ���E	~3���z�A��K*

A2S/average_reward_1\��Cp��,       ���E	��M��z�A��K*

A2S/average_reward_1H��CO7{,       ���E	쒚��z�A��K*

A2S/average_reward_1ͬ�Cy]b�,       ���E	�9{��z�A��K*

A2S/average_reward_1�5�Co���,       ���E	�����z�A��K*

A2S/average_reward_1 ��Cb��c,       ���E	D����z�A��K*

A2S/average_reward_13S�C���,       ���E	ێ���z�A��K*

A2S/average_reward_1׃�C��,       ���E	lx��z�A��K*

A2S/average_reward_1���C\ih,       ���E	9�e��z�A��K*

A2S/average_reward_1
w�C~�i�,       ���E	+T���z�A��K*

A2S/average_reward_1���C�\�,       ���E	�>��z�A��K*

A2S/average_reward_1���C��
N,       ���E	����z�A��K*

A2S/average_reward_1��C�&Q,       ���E	~���z�A��K*

A2S/average_reward_1f�C�Z�,       ���E	�Q$��z�A��K*

A2S/average_reward_1
��C�
�,       ���E	�b!��z�A��K*

A2S/average_reward_1���Cn8�G,       ���E	0���z�A��K*

A2S/average_reward_1H��Cq,       ���E	�;e��z�A�L*

A2S/average_reward_1�H�C�?N�,       ���E	+T���z�A��L*

A2S/average_reward_1�>�C�cX,       ���E	�"���z�A��L*

A2S/average_reward_1f&�C����,       ���E	�)��z�A��L*

A2S/average_reward_1��C�1�,       ���E	Rê��z�A��L*

A2S/average_reward_13��Cި|6,       ���E	�����z�A�L*

A2S/average_reward_1���CA�!o,       ���E	m4y��z�A��L*

A2S/average_reward_1R��C=�x,       ���E	͘���z�A�L*

A2S/average_reward_1���C>'�,       ���E	rD���z�A�L*

A2S/average_reward_1�L�C��,       ���E	3����z�A��L*

A2S/average_reward_1 ��C;��,       ���E	Q����z�A��L*

A2S/average_reward_1q=�C�#�,       ���E	^����z�A��L*

A2S/average_reward_1q�CY,͌,       ���E	 �.��z�A�L*

A2S/average_reward_1��C}��*,       ���E	n����z�A�L*

A2S/average_reward_1\��CN��,       ���E	�Y���z�A��L*

A2S/average_reward_1)\�C^��,       ���E	�\���z�A��L*

A2S/average_reward_1�Q�CB?,       ���E	���z�A��L*

A2S/average_reward_1{t�Cie��,       ���E	�LE��z�A�L*

A2S/average_reward_1�c�C��M,       ���E	<���z�AܭL*

A2S/average_reward_1�C(n�M,       ���E	*>��z�A��L*

A2S/average_reward_1ŸC�^j�,       ���E	�5G��z�A�L*

A2S/average_reward_1
W�C���,       ���E	W�M��z�A��L*

A2S/average_reward_1�Z�C7F�,       ���E	�0���z�A�L*

A2S/average_reward_1qݯC�(yx,       ���E	����z�AԸL*

A2S/average_reward_1͌�C��j},       ���E	�{���z�A��L*

A2S/average_reward_1RذC�"�,       ���E	�Z���z�A��L*

A2S/average_reward_1\��C����,       ���E	[Uq��z�A��L*

A2S/average_reward_1�Y�C��٦,       ���E	�����z�A��L*

A2S/average_reward_133�Cv�?�,       ���E	W]���z�A��L*

A2S/average_reward_133�C��,       ���E	1���z�A��L*

A2S/average_reward_133�C�nm,       ���E	�HF��z�A��L*

A2S/average_reward_1{4�Ck��n,       ���E	й���z�A��L*

A2S/average_reward_1��C��C,       ���E	�|��z�A��L*

A2S/average_reward_1 @�C��i,       ���E	����z�A��L*

A2S/average_reward_1R��C�'��,       ���E	�����z�A��L*

A2S/average_reward_13S�Cw�e�,       ���E	��W��z�A��L*

A2S/average_reward_1�>�C�5�,       ���E	|X �z�A��L*

A2S/average_reward_1f�CD�i,       ���E	�/�z�A��L*

A2S/average_reward_1��C�h�,       ���E	ݱ��z�A��L*

A2S/average_reward_1{T�C����,       ���E	����z�A؆M*

A2S/average_reward_1���C#䍘,       ���E	�г�z�A�M*

A2S/average_reward_1ף�C��,       ���E	�z~�z�A�M*

A2S/average_reward_1���C[�C,       ���E	[��z�A��M*

A2S/average_reward_1\ϻC
��=,       ���E	���z�A��M*

A2S/average_reward_1���C��ڥ,       ���E	E&[	�z�A��M*

A2S/average_reward_1)\�C��5,       ���E	g+t	�z�AM*

A2S/average_reward_1)��C� ?H,       ���E	֡a�z�A��M*

A2S/average_reward_1�p�CX� ,       ���E	����z�A��M*

A2S/average_reward_1�Q�C��^v,       ���E	����z�A��M*

A2S/average_reward_1
7�CN��,       ���E	����z�A�M*

A2S/average_reward_1q��CI��L,       ���E	⡈�z�AͨM*

A2S/average_reward_1�l�Cx/g],       ���E	�p��z�A�M*

A2S/average_reward_1�зCp���,       ���E	�/S�z�AϯM*

A2S/average_reward_1R�CU�)�,       ���E	��z�A�M*

A2S/average_reward_1q��C�f�c,       ���E	(=�z�A��M*

A2S/average_reward_1N�C��R&,       ���E	�J��z�AϴM*

A2S/average_reward_1q��Ci���,       ���E		���z�A�M*

A2S/average_reward_1�+�C�9�,       ���E	���z�AͽM*

A2S/average_reward_1\�CǱ�,       ���E	@5��z�A��M*

A2S/average_reward_1)��C��� ,       ���E	�a��z�A��M*

A2S/average_reward_1f��Cc#�u,       ���E	��c�z�A��M*

A2S/average_reward_1H!�C����,       ���E	�M�z�A��M*

A2S/average_reward_1���Cx	Y,       ���E	���z�A��M*

A2S/average_reward_1���CNޭW,       ���E	�:�z�A��M*

A2S/average_reward_1
��CF��,       ���E	� K�z�A��M*

A2S/average_reward_1���C}�{u,       ���E	��o�z�A��M*

A2S/average_reward_1�:�C98[H,       ���E	����z�A��M*

A2S/average_reward_1Rx�C|;9F,       ���E	\O��z�A��M*

A2S/average_reward_1���C�� ,       ���E	C�z�A��M*

A2S/average_reward_1f��C�^�f,       ���E	s��z�A��M*

A2S/average_reward_1R�CY �,,       ���E	Ņ��z�A��M*

A2S/average_reward_1�U�CaH\,       ���E	;w��z�A��M*

A2S/average_reward_1ף�C�~�,       ���E	��C�z�A��M*

A2S/average_reward_1�(�C
e�<,       ���E	��s �z�A��M*

A2S/average_reward_1 ��C�̓S,       ���E	��� �z�A��M*

A2S/average_reward_1���CCW��,       ���E	4�"!�z�A��M*

A2S/average_reward_1���C\�ֆ,       ���E	���"�z�A��M*

A2S/average_reward_1�#�C�oE,       ���E	r�$�z�AʂN*

A2S/average_reward_1���C����,       ���E	H&|&�z�A��N*

A2S/average_reward_1�C�>��,       ���E	��=(�z�A��N*

A2S/average_reward_1���Cb�E,       ���E	��c(�z�A͐N*

A2S/average_reward_1{��C�u�,       ���E	�[)�z�A��N*

A2S/average_reward_1\�C���,       ���E	w�)�z�A��N*

A2S/average_reward_1H!�C�@,       ���E	���)�z�A��N*

A2S/average_reward_1���C����,       ���E	�W�)�z�A��N*

A2S/average_reward_1���C'��,       ���E	���*�z�A��N*

A2S/average_reward_1���C�4�y,       ���E	^��*�z�A�N*

A2S/average_reward_1
w�C��,       ���E	���+�z�A��N*

A2S/average_reward_1.�Cg�r�,       ���E	W-�z�A��N*

A2S/average_reward_1���C?�,       ���E	4�Q-�z�A��N*

A2S/average_reward_1���C�G�l,       ���E	4�p.�z�AɨN*

A2S/average_reward_1q=�C���,       ���E	cZ�.�z�AӪN*

A2S/average_reward_1��CΪ+�,       ���E	�|/�z�A�N*

A2S/average_reward_1�z�C
���,       ���E	6��/�z�A��N*

A2S/average_reward_1��C��8�,       ���E	�0�z�AױN*

A2S/average_reward_1
��C�I��       �v@�	1��1�z�AױN*�

A2S/kl�#?

A2S/average_advantage�8=

A2S/policy_network_loss:}y?

A2S/value_network_lossT��C

A2S/q_network_loss�1�C�_�R