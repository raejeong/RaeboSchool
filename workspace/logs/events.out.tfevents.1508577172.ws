       �K"	   ��z�Abrain.Event:2WYD��;     ��_�	e���z�A"��
s
A2S/observationsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
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
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
VariableV2*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
	container *
shape:*
dtype0
�
<A2S/backup_policy_network/backup_policy_network/fc0/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/bGA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b
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
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta
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
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
validate_shape(
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
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
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
3A2S/backup_policy_network/LayerNorm/batchnorm/RsqrtRsqrt1A2S/backup_policy_network/LayerNorm/batchnorm/add*'
_output_shapes
:���������*
T0
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
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
�
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:*
T0
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
<A2S/backup_policy_network/backup_policy_network/out/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/wPA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:*
T0
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
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  ��
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
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
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
VariableV2*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:*
dtype0
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
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
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
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
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
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:*
T0
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
;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
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
1A2S/best_policy_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
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
)A2S/best_policy_network/dropout/keep_probConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
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
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
�
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
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
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
�
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
�
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
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
9A2S/backup_value_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
valueB*  �?
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
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(
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
EA2S/backup_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
w
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼�+
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
A2S/backup_value_network/AbsAbs2A2S/backup_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:���������
e
 A2S/backup_value_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *���>
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
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *��̽*
dtype0
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
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: 
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
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
	container *
shape
:*
dtype0
�
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
8A2S/backup_value_network/backup_value_network/out/w/readIdentity3A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
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
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0
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
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
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
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
�
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
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
:���������*

Tidx0*
	keep_dims(
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
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
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
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
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
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
q
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
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
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:���������
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
A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
'A2S/Normal_2/batch_shape_tensor/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
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
N*
_output_shapes
:*

Tidx0*
T0
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
 :������������������*
seed2�*

seed
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*4
_output_shapes"
 :������������������*
T0
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
A2S/addAddA2S/mulA2S/Normal_1/loc*
T0*4
_output_shapes"
 :������������������
h
A2S/Reshape_2/shapeConst*!
valueB"����      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������
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
LA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB"      
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
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
�
=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zerosConst*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
valueB*    *
dtype0
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
(A2S/backup_q_network/LayerNorm/beta/readIdentity#A2S/backup_q_network/LayerNorm/beta*
_output_shapes
:*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta
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
A2S/backup_q_network/add_1AddA2S/backup_q_network/mulA2S/backup_q_network/mul_1*'
_output_shapes
:���������*
T0
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
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
TA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shape*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
	container *
shape:
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
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  �?
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
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
"A2S/best_q_network/LayerNorm/gamma
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
)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
'A2S/best_q_network/LayerNorm/gamma/readIdentity"A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
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
*A2S/best_q_network/LayerNorm/batchnorm/addAdd-A2S/best_q_network/LayerNorm/moments/variance,A2S/best_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:���������*
T0
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
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/add_1,A2S/best_q_network/best_q_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/out/b/read*'
_output_shapes
:���������*
T0
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
A2S/Const_4Const*
_output_shapes
:*
valueB"       *
dtype0
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

A2S/Mean_3MeanA2S/SquaredDifferenceA2S/Const_5*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
%A2S/gradients/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_2_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
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
#A2S/gradients/A2S/Mean_2_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
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
"A2S/gradients/A2S/Mean_2_grad/CastCast&A2S/gradients/A2S/Mean_2_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
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
 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_2_grad/truedivA2S/advantages*'
_output_shapes
:���������*
T0
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
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ShapeShapeA2S/Normal_3/log_prob/mul*
_output_shapes
:*
T0*
out_type0
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
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape
�
GA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1*
_output_shapes
: *
T0
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
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1MulA2S/Normal_3/log_prob/mul/xEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
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
GA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
�
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *   @*
dtype0
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
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_3/log_prob/standardize/sub*
out_type0*
_output_shapes
:*
T0
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*'
_output_shapes
:���������*
T0
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
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1
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
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
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
KA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
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
:*

Tidx0*
	keep_dims( 
�
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul.A2S/best_policy_network/LayerNorm/moments/mean]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
A2S/gradients/AddN_1AddN_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
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
PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0
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
UA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
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
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
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
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeRangeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0
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
: *

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
LA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1C^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
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
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
�
A2S/beta1_power/readIdentityA2S/beta1_power*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
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
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:*
T0
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
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
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
4A2S/A2S/best_policy_network/LayerNorm/beta/Adam/readIdentity/A2S/A2S/best_policy_network/LayerNorm/beta/Adam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
�
CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0
�
1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1
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
VariableV2*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0
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
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
�
DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    
�
2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1
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
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/readIdentity2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
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
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
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
LA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    
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
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
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
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
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
@A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam&A2S/best_policy_network/LayerNorm/beta/A2S/A2S/best_policy_network/LayerNorm/beta/Adam1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
'A2S/gradients_1/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_3_grad/Prod_1)A2S/gradients_1/A2S/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
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
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*'
_output_shapes
:���������*
T0
�
0A2S/gradients_1/A2S/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add_2*
T0*
out_type0*
_output_shapes
:
}
2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1ShapeA2S/returns*
_output_shapes
:*
T0*
out_type0
�
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
�
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:���������
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
5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
5A2S/gradients_1/A2S/best_value_network/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
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
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
out_type0*
_output_shapes
:*
T0
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
N*'
_output_shapes
:���������*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
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
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addAddCA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*
T0*0
_output_shapes
:������������������*

Tmultiples0
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
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
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
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeRangeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/delta*

Tidx0*
_output_shapes
:
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
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
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumIA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0
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
7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
A2S/beta1_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
A2S/beta1_power_1
VariableV2*
_output_shapes
: *
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape: *
dtype0
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
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: *
T0
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
VariableV2*
_output_shapes
: *
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container *
shape: *
dtype0
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
AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
�
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
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
shape:*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container 
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
CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0
�
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1
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
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:*
T0
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
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
�
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
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
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
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
%A2S/gradients_2/A2S/Mean_4_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
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
7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
CA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
:*

Tidx0*
	keep_dims( *
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
�
ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape
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
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
Tshape0*
_output_shapes
:*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*
T0*'
_output_shapes
:���������
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape
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
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*
_output_shapes
:
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
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:���������*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/CastCastMA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
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
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
out_type0*
_output_shapes
:*
T0
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
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
dA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*f
_class\
ZXloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape
�
fA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*b
_classX
VTloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������*
T0
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
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/add_grad/Shape3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
?A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul8^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
�
GA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul
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
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container 
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
DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    
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
<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
*A2S/A2S/best_q_network/LayerNorm/beta/Adam
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
DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    
�
2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1
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
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
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
A2S/Adam_2/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( 
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
=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( 
�
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
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
A2S/average_reward_1ScalarSummaryA2S/average_reward_1/tagsA2S/average_reward*
T0*
_output_shapes
: "���ῗ     (+�E	X��z�AJ��
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
A2S/actionsPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
q
A2S/advantagesPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
V
A2S/learning_ratePlaceholder*
_output_shapes
:*
shape:*
dtype0
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
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
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
_output_shapes

:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
	container *
shape
:*
dtype0
�
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
 A2S/backup_policy_network/MatMulMatMulA2S/observations:A2S/backup_policy_network/backup_policy_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
A2S/backup_policy_network/addAdd A2S/backup_policy_network/MatMul:A2S/backup_policy_network/backup_policy_network/fc0/b/read*'
_output_shapes
:���������*
T0
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
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta
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
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
�
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0
�
BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
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
3A2S/backup_policy_network/LayerNorm/batchnorm/RsqrtRsqrt1A2S/backup_policy_network/LayerNorm/batchnorm/add*'
_output_shapes
:���������*
T0
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
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
�
<A2S/backup_policy_network/backup_policy_network/out/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/wPA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
:A2S/backup_policy_network/backup_policy_network/out/w/readIdentity5A2S/backup_policy_network/backup_policy_network/out/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
�
GA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zerosConst*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
valueB*    *
dtype0
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
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b
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
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  ��*
dtype0
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
seed2H*
dtype0*
_output_shapes

:*

seed*
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
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:*
T0
�
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
6A2S/best_policy_network/LayerNorm/moments/StopGradientStopGradient.A2S/best_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
�
;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:���������
�
DA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
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
A2S/best_policy_network/mulMulA2S/best_policy_network/mul/x1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:���������*
T0
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
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      *
dtype0
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
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(
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
EA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
valueB*    
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
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:*
T0
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
7A2S/backup_value_network/LayerNorm/moments/StopGradientStopGradient/A2S/backup_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:���������
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
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼�+*
dtype0
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
 A2S/backup_value_network/mul_1/xConst*
_output_shapes
: *
valueB
 *���>*
dtype0
�
A2S/backup_value_network/mul_1Mul A2S/backup_value_network/mul_1/xA2S/backup_value_network/Abs*'
_output_shapes
:���������*
T0
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
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *��̽*
dtype0
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
\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shape*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2�*
dtype0*
_output_shapes

:*

seed
�
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: 
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
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
	container *
shape
:*
dtype0
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
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(*
_output_shapes
:
�
8A2S/backup_value_network/backup_value_network/out/b/readIdentity3A2S/backup_value_network/backup_value_network/out/b*
_output_shapes
:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b
�
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  ��*
dtype0
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
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0
�
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
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
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
�
%A2S/best_value_network/LayerNorm/beta
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
,A2S/best_value_network/LayerNorm/beta/AssignAssign%A2S/best_value_network/LayerNorm/beta7A2S/best_value_network/LayerNorm/beta/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(
�
*A2S/best_value_network/LayerNorm/beta/readIdentity%A2S/best_value_network/LayerNorm/beta*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
�
7A2S/best_value_network/LayerNorm/gamma/Initializer/onesConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*  �?*
dtype0
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
?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
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
A2S/best_value_network/mul/xConst*
_output_shapes
: *
valueB
 *��?*
dtype0
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
A2S/best_value_network/add_1AddA2S/best_value_network/mulA2S/best_value_network/mul_1*'
_output_shapes
:���������*
T0
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
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
�
/A2S/best_value_network/best_value_network/out/w
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
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
�
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:*
T0
�
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
A2S/ReshapeReshapeA2S/backup_policy_network/add_2A2S/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
d
A2S/Reshape_1/shapeConst*
_output_shapes
:*
valueB"����   *
dtype0
�
A2S/Reshape_1ReshapeA2S/best_policy_network/add_2A2S/Reshape_1/shape*'
_output_shapes
:���������*
T0*
Tshape0
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
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0
�
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:���������
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
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
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

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_2/batch_shape_tensor/BroadcastArgsA2S/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
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
A2S/concat*
dtype0*4
_output_shapes"
 :������������������*
seed2�*

seed*
T0
�
A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*4
_output_shapes"
 :������������������*
T0
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
A2S/Reshape_2/shapeConst*!
valueB"����      *
dtype0*
_output_shapes
:
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
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
TA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
seed2�*
dtype0*
_output_shapes

:*

seed
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes
: 
�
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
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
VariableV2*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
2A2S/backup_q_network/backup_q_network/fc0/b/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/b=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zeros*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
0A2S/backup_q_network/backup_q_network/fc0/b/readIdentity+A2S/backup_q_network/backup_q_network/fc0/b*
_output_shapes
:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b
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
*A2S/backup_q_network/LayerNorm/beta/AssignAssign#A2S/backup_q_network/LayerNorm/beta5A2S/backup_q_network/LayerNorm/beta/Initializer/zeros*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
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
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma
�
=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
3A2S/backup_q_network/LayerNorm/moments/StopGradientStopGradient+A2S/backup_q_network/LayerNorm/moments/mean*'
_output_shapes
:���������*
T0
�
8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_q_network/add3A2S/backup_q_network/LayerNorm/moments/StopGradient*'
_output_shapes
:���������*
T0
�
AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
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
A2S/backup_q_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *���>
�
A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*'
_output_shapes
:���������*
T0
�
A2S/backup_q_network/add_1AddA2S/backup_q_network/mulA2S/backup_q_network/mul_1*'
_output_shapes
:���������*
T0
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

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
seed2�*
dtype0*
_output_shapes

:
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
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  ��
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
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
�
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
�
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0
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
VariableV2*
_output_shapes
:*
shared_name *4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
	container *
shape:*
dtype0
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
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  �?
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
)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(
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
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(
�
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
�
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/add_1,A2S/best_q_network/best_q_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/out/b/read*'
_output_shapes
:���������*
T0
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
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
T0*
_output_shapes
: 
`
A2S/Normal_3/log_prob/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *�?k?
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

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
t
A2S/policy_network_loss/tagsConst*
dtype0*
_output_shapes
: *(
valueB BA2S/policy_network_loss
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
#A2S/gradients/A2S/Mean_2_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
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
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
BA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape
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
6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
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
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
T0*
out_type0*
_output_shapes
:
�
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
6A2S/gradients/A2S/best_policy_network/add_1_grad/ShapeShapeA2S/best_policy_network/mul*
out_type0*
_output_shapes
:*
T0
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1ShapeA2S/best_policy_network/mul_1*
out_type0*
_output_shapes
:*
T0
�
FA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
:*

Tidx0*
	keep_dims( 
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape.A2S/best_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
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
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
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
[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape
�
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:*
T0
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeConst*
_output_shapes
: *
value	B :*
dtype0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addAddDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/CastCastPA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
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
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape6A2S/best_policy_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
�
dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
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
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeReshape]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
�
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileTileIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0
�
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape.A2S/best_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
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
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*
dtype0
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
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1*
_output_shapes
:
�
8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
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
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
�
A2S/beta1_power/readIdentityA2S/beta1_power*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: *
T0
�
A2S/beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0
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
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
�
A2S/beta2_power/readIdentityA2S/beta2_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
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
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/w/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
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
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/b/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
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
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    
�
0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam
VariableV2*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0
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
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
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
A2S/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
S
A2S/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
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
@A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam&A2S/best_policy_network/LayerNorm/beta/A2S/A2S/best_policy_network/LayerNorm/beta/Adam1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/b:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonKA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
use_nesterov( 
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
'A2S/gradients_1/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
z
%A2S/gradients_1/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
'A2S/gradients_1/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
CA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIdentity2A2S/gradients_1/A2S/SquaredDifference_grad/Reshape<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*E
_class;
97loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Reshape*'
_output_shapes
:���������*
T0
�
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg
�
7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
out_type0*
_output_shapes
:*
T0
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
5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_1SumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
_output_shapes
:*
T0*
out_type0
�
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
�
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������*
T0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
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
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:���������*
T0
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
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*'
_output_shapes
:���������*
T0
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
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0
�
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:���������
�
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*
T0*0
_output_shapes
:������������������*

Tmultiples0
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
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
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
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegYA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
`A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpX^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeT^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
�
hA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshapea^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*j
_class`
^\loc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������*
T0
�
jA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitySA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Nega^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*f
_class\
ZXloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:���������*
T0
�
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
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
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/MaximumMaximumPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/y*#
_output_shapes
:���������*
T0
�
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
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
:*

Tidx0*
	keep_dims( 
�
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
�
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
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
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
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
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0
�
8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam
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
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/b/AdamJA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
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
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
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
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/AssignAssign/A2S/A2S/best_value_network/LayerNorm/gamma/AdamAA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(
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
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
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
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
BA2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&A2S/best_value_network/LayerNorm/gamma/A2S/A2S/best_value_network/LayerNorm/gamma/Adam1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
use_nesterov( 
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
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
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
A2S/gradients_2/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
A2S/gradients_2/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
k
A2S/gradients_2/FillFillA2S/gradients_2/ShapeA2S/gradients_2/Const*
_output_shapes
: *
T0
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
'A2S/gradients_2/A2S/Mean_4_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
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
'A2S/gradients_2/A2S/Mean_4_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_4_grad/Prod_1)A2S/gradients_2/A2S/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0
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
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
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
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
CA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
3A2S/gradients_2/A2S/best_q_network/add_1_grad/ShapeShapeA2S/best_q_network/mul*
T0*
out_type0*
_output_shapes
:
�
5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1ShapeA2S/best_q_network/mul_1*
_output_shapes
:*
T0*
out_type0
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
5A2S/gradients_2/A2S/best_q_network/add_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_1SumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1MulA2S/best_q_network/mul/xFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
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
\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul)A2S/best_q_network/LayerNorm/moments/meanZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
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
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:���������
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*'
_output_shapes
:���������*
T0
�
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
�
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0
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
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
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
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradientK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:���������*
T0
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
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
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
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
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
: *

Tidx0*
	keep_dims( 
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
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*
dtype0
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
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*
N*'
_output_shapes
:���������*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape
�
1A2S/gradients_2/A2S/best_q_network/add_grad/ShapeShapeA2S/best_q_network/MatMul*
_output_shapes
:*
T0*
out_type0
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
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
A2S/beta1_power_2/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
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
9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    
�
-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1
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
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(
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
VariableV2*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
7A2S/A2S/best_q_network/best_q_network/out/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/w/AdamBA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
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
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
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
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/b0A2S/A2S/best_q_network/best_q_network/out/b/Adam2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonHA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
�
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
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
A2S/average_reward_1/tagsConst*
_output_shapes
: *%
valueB BA2S/average_reward_1*
dtype0
u
A2S/average_reward_1ScalarSummaryA2S/average_reward_1/tagsA2S/average_reward*
T0*
_output_shapes
: ""0
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
A2S/average_reward_1:0;/*       ����	?&F��z�A*

A2S/average_reward_1  @A��։*       ����	�J��z�A*

A2S/average_reward_1  HAb�5�*       ����	��N��z�A,*

A2S/average_reward_1��jA�Rj�*       ����	�S��z�A<*

A2S/average_reward_1  pA�Yʚ*       ����	}-Z��z�AO*

A2S/average_reward_1��|A���*       ����	_�\��z�AX*

A2S/average_reward_1��jALp�*       ����	w�_��z�Ad*

A2S/average_reward_1I�dA�d5�*       ����	�nc��z�Ao*

A2S/average_reward_1  ^A�±*       ����	��f��z�Ay*

A2S/average_reward_1rWA�y+       ��K	��h��z�A�*

A2S/average_reward_1ffNA�X |+       ��K	�k��z�A�*

A2S/average_reward_1�EGAgF��+       ��K	��n��z�A�*

A2S/average_reward_1UUIA=N��+       ��K	,�r��z�A�*

A2S/average_reward_1�NLAN��+       ��K	\�v��z�A�*

A2S/average_reward_1۶MA�?�_+       ��K	z��z�A�*

A2S/average_reward_1��NA�[{�+       ��K	Ƴ}��z�A�*

A2S/average_reward_1  NAa���+       ��K	�����z�A�*

A2S/average_reward_1NA��	�+       ��K	�a���z�A�*

A2S/average_reward_1�8NA��Y+       ��K	߬���z�A�*

A2S/average_reward_1  PA���+       ��K	����z�A�*

A2S/average_reward_1��LAW�+       ��K	DI���z�A�*

A2S/average_reward_1%IRA�O#%+       ��K	���z�A�*

A2S/average_reward_1�SA�9c�+       ��K	'����z�A�*

A2S/average_reward_1��UA=�A+       ��K	�ƚ��z�A�*

A2S/average_reward_1��RA}���+       ��K	j���z�A�*

A2S/average_reward_1�UA\]�+       ��K	G���z�A�*

A2S/average_reward_1  XAg�(+       ��K	�-���z�A�*

A2S/average_reward_1�%TA��s�+       ��K	�����z�A�*

A2S/average_reward_1%IRA�<�+       ��K	�Ӭ��z�A�*

A2S/average_reward_1�iTA���c+       ��K	F���z�A�*

A2S/average_reward_1��SA�=UU�       L	vM	~���z�A�*�

A2S/kl���<

A2S/average_advantageEPܾ

A2S/policy_network_loss�"�

A2S/value_network_loss��%@

A2S/q_network_loss���?�M��+       ��K	�~��z�A�*

A2S/average_reward_1�WA�o��+       ��K	��z�A�*

A2S/average_reward_1 �WAw�j+       ��K	Ĺ	��z�A�*

A2S/average_reward_1�MVA�Co�+       ��K	Ŀ��z�A�*

A2S/average_reward_1��YAײ�+       ��K	\���z�A�*

A2S/average_reward_1�ZA�̳+       ��K	����z�A�*

A2S/average_reward_1�]ALG�+       ��K	F���z�A�*

A2S/average_reward_1L�_A�� +       ��K	0_%��z�A�*

A2S/average_reward_1�k`A]|R*+       ��K	��)��z�A�*

A2S/average_reward_1  `A͍�+       ��K	�Q-��z�A�*

A2S/average_reward_1��^A�D�+       ��K	�P2��z�A�*

A2S/average_reward_1dp^A�D�+       ��K	1�7��z�A�*

A2S/average_reward_1n�^A*�s�+       ��K	H;��z�A�*

A2S/average_reward_1w�]AҪ�v+       ��K	��B��z�A�*

A2S/average_reward_1  `A�	O+       ��K	�J��z�A�*

A2S/average_reward_1�aA��+       ��K	�P��z�A�*

A2S/average_reward_1ozcAl*�_+       ��K	��T��z�A�*

A2S/average_reward_1WcAc�2�+       ��K	�vY��z�A�*

A2S/average_reward_1��bAC��+       ��K	1�]��z�A�*

A2S/average_reward_1��aAó�K+       ��K	!Fe��z�A�*

A2S/average_reward_1�cA +       ��K	�n��z�A�*

A2S/average_reward_1��fA%:�+       ��K	M�s��z�A�*

A2S/average_reward_1�gAЎ+       ��K	?�y��z�A�*

A2S/average_reward_15�hAU���+       ��K	����z�A�*

A2S/average_reward_1�^jA���+       ��K	����z�A�*

A2S/average_reward_1~�jA�J�+       ��K	�!���z�A�*

A2S/average_reward_1I�jAmA+       ��K	���z�A�*

A2S/average_reward_1��jA�.F�+       ��K	�O���z�A�*

A2S/average_reward_1�kA<KR�+       ��K	�:���z�A�*

A2S/average_reward_1�lA�b\+       ��K	K����z�A�*

A2S/average_reward_1��mA�6ɖ       L	vM	�����z�A�*�

A2S/klØ�>

A2S/average_advantageI��

A2S/policy_network_loss��i�

A2S/value_network_lossb~{?

A2S/q_network_lossd9�?�eY�+       ��K	�]���z�A�*

A2S/average_reward_1mnAa��+       ��K	����z�A�*

A2S/average_reward_1�oA^��+       ��K	�2���z�A�*

A2S/average_reward_1��zAD���+       ��K	�3��z�A�*

A2S/average_reward_1 �~A���+       ��K	����z�A�*

A2S/average_reward_1C/�A��f+       ��K	)��z�A�*

A2S/average_reward_1UU�AJD3`+       ��K	Ƈ.��z�A�*

A2S/average_reward_1=P�A�7��+       ��K	�=��z�A�	*

A2S/average_reward_1���Ab�+       ��K		�B��z�A�	*

A2S/average_reward_1l�A��+       ��K	��G��z�A�	*

A2S/average_reward_1ff�AP��+       ��K	�uX��z�A�
*

A2S/average_reward_1�D�A�~+       ��K	��^��z�A�
*

A2S/average_reward_1ǓA0�c>+       ��K	��b��z�A�
*

A2S/average_reward_1�e�AWR�+       ��K	�g��z�A�
*

A2S/average_reward_1��A.%�\+       ��K	�Hk��z�A�
*

A2S/average_reward_1���Av�j+       ��K	�{o��z�A�
*

A2S/average_reward_1�k�AbꞬ+       ��K	�Kt��z�A�*

A2S/average_reward_1�c�A�7��+       ��K	5�x��z�A�*

A2S/average_reward_1'v�Ao3h@+       ��K	�b}��z�A�*

A2S/average_reward_1AT�A��&h+       ��K	�܁��z�A�*

A2S/average_reward_133�A��׊+       ��K	=����z�A�*

A2S/average_reward_1?,�AQ�v�+       ��K	�֋��z�A�*

A2S/average_reward_1p>�Am�\I+       ��K	�%���z�A�*

A2S/average_reward_1>�A7$��+       ��K	�����z�A�*

A2S/average_reward_1��A���+       ��K	�z���z�A�*

A2S/average_reward_1�Av�$#+       ��K	4���z�A�*

A2S/average_reward_1<�A�Syc+       ��K	�����z�A�*

A2S/average_reward_1J�A�2ˁ+       ��K	����z�A�*

A2S/average_reward_1�.�A�a�+       ��K	�s���z�A�*

A2S/average_reward_1_��A1�a�+       ��K	s����z�A�*

A2S/average_reward_1���A��b�       L	vM	���z�A�*�

A2S/kl,e#>

A2S/average_advantage����

A2S/policy_network_loss�_��

A2S/value_network_loss�#�C

A2S/q_network_loss,7�Cd�Щ+       ��K	�9��z�A�*

A2S/average_reward_1���A�d��+       ��K	j���z�A�*

A2S/average_reward_1���A��+       ��K	�&��z�A�*

A2S/average_reward_1,��A�Z�+       ��K	��,��z�A�*

A2S/average_reward_1��A�N��+       ��K	�2��z�A�*

A2S/average_reward_1��A�!C�+       ��K	�U8��z�A�*

A2S/average_reward_1 ��A�� �+       ��K	Y,>��z�A�*

A2S/average_reward_1�,�A�FA+       ��K	+zC��z�A�*

A2S/average_reward_1
^�A-{+       ��K	�CI��z�A�*

A2S/average_reward_1�~�As^y�+       ��K	�M��z�A�*

A2S/average_reward_1�Q�A���+       ��K	�4R��z�A�*

A2S/average_reward_1ף�A��_+       ��K	~SU��z�A�*

A2S/average_reward_1�z�A
2�5+       ��K	�i_��z�A�*

A2S/average_reward_1  �A���+       ��K	u�b��z�A�*

A2S/average_reward_1��A�s+       ��K	_r���z�A�*

A2S/average_reward_1���A;�+�+       ��K	�G���z�A�*

A2S/average_reward_1���Af�U+       ��K	?����z�A�*

A2S/average_reward_1q=�A����+       ��K	�G���z�A�*

A2S/average_reward_133�A����+       ��K	�Η��z�A�*

A2S/average_reward_1���AWc��+       ��K	Q ���z�A�*

A2S/average_reward_1\��Aw+       ��K	,ޠ��z�A�*

A2S/average_reward_1)\�A��+       ��K	 ����z�A�*

A2S/average_reward_1��A�͂+       ��K	Z���z�A�*

A2S/average_reward_1���At��+       ��K	5j���z�A�*

A2S/average_reward_1��A�p��+       ��K	p���z�A�*

A2S/average_reward_1\��A�b�D+       ��K	?����z�A�*

A2S/average_reward_1�G�A�V'+       ��K	�R���z�A�*

A2S/average_reward_1��A����+       ��K	O���z�A�*

A2S/average_reward_1���A��uV+       ��K	v����z�A�*

A2S/average_reward_1H��Ay���+       ��K	:���z�A�*

A2S/average_reward_1ff�A$��*�       L	vM	���z�A�*�

A2S/klJ?

A2S/average_advantaget�?

A2S/policy_network_loss�Z�>

A2S/value_network_loss4TxA

A2S/q_network_lossIk5A��v+       ��K	t���z�A�*

A2S/average_reward_1R��A]�+       ��K	�19��z�A�*

A2S/average_reward_1ף�Aq�ɹ+       ��K	`*@��z�A�*

A2S/average_reward_1�G�AI�1�+       ��K	��C��z�A�*

A2S/average_reward_1)\�A=��d+       ��K	0�q��z�A�*

A2S/average_reward_1�B��a+       ��K	�����z�A�*

A2S/average_reward_1�B�'�+       ��K	�����z�A�*

A2S/average_reward_1�zB1B9+       ��K	����z�A�*

A2S/average_reward_1
�
B����+       ��K	����z�A�*

A2S/average_reward_1�B��+       ��K	�B!��z�A� *

A2S/average_reward_1�B�y�+       ��K	ZH'��z�A� *

A2S/average_reward_1�B*zњ+       ��K	��X��z�A�"*

A2S/average_reward_1�GB0�9(+       ��K	�8���z�A�#*

A2S/average_reward_1�#B��	+       ��K	���z�A�#*

A2S/average_reward_1\�#BSԼF+       ��K	����z�A�#*

A2S/average_reward_1��#B�\�[+       ��K	����z�A�$*

A2S/average_reward_1)\(B+���+       ��K	p����z�A�%*

A2S/average_reward_1{-B����+       ��K	ذ���z�A�&*

A2S/average_reward_1��0B���+       ��K	y0���z�A�&*

A2S/average_reward_1R�0B�|�S+       ��K	�����z�A�&*

A2S/average_reward_1��0B#���+       ��K	�/��z�A�(*

A2S/average_reward_1)\7B�rj�+       ��K	�B@��z�A�(*

A2S/average_reward_1��8B>.+       ��K	�I��z�A�(*

A2S/average_reward_1�z9B�n�+       ��K	VAv��z�A�**

A2S/average_reward_1  ?B����+       ��K	�|��z�A�**

A2S/average_reward_1�?B����+       ��K	�M��z�A�**

A2S/average_reward_1
�>BpN=�+       ��K	:����z�A�**

A2S/average_reward_1�G?B4�O�+       ��K	����z�A�**

A2S/average_reward_1�@B��r�+       ��K	u����z�A�**

A2S/average_reward_1�@B=a�?+       ��K	Mn���z�A�+*

A2S/average_reward_1�CB!�U+       ��K	]����z�A�,*

A2S/average_reward_1=
FBW]*+       ��K	����z�A�,*

A2S/average_reward_1��EB�i�4+       ��K	"����z�A�,*

A2S/average_reward_1�(GB��r	+       ��K	5����z�A�-*

A2S/average_reward_1�GB�W�+       ��K	�.���z�A�-*

A2S/average_reward_1�GB_zwk+       ��K	S���z�A�-*

A2S/average_reward_1ffHB�`B+       ��K	���z�A�.*

A2S/average_reward_1�NB�U�+       ��K	� ��z�A�.*

A2S/average_reward_1ffNB�.�a+       ��K	2�L��z�A�0*

A2S/average_reward_1�SBjP�d+       ��K	剅��z�A�1*

A2S/average_reward_1ףYB����       L	vM	J����z�A�1*�

A2S/kl~��>

A2S/average_advantage󷌿

A2S/policy_network_loss\� �

A2S/value_network_loss��`B

A2S/q_network_loss�&VB"�](+       ��K	�3���z�A�1*

A2S/average_reward_1�pYBk��+       ��K	9����z�A�1*

A2S/average_reward_1{ZB�#+       ��K	cH��z�A�3*

A2S/average_reward_1R�]B� J3+       ��K	 l	��z�A�3*

A2S/average_reward_1�(]B���+       ��K	c�'��z�A�4*

A2S/average_reward_1�z_B��$�+       ��K	7҆��z�A�6*

A2S/average_reward_1�QjBk�?�+       ��K	 ܑ��z�A�6*

A2S/average_reward_1��jB��+       ��K	
���z�A�6*

A2S/average_reward_1  iB���+       ��K	����z�A�7*

A2S/average_reward_1��hBǌ��+       ��K	{���z�A�8*

A2S/average_reward_1  mBs3 ?+       ��K	�t���z�A�9*

A2S/average_reward_1)\pB���	+       ��K	*��z�A�9*

A2S/average_reward_1R�rBzō+       ��K	����z�A�9*

A2S/average_reward_1  sB�)�+       ��K	�T��z�A�:*

A2S/average_reward_1R�rB�a�M+       ��K	p?<��z�A�;*

A2S/average_reward_1  xB��`�+       ��K	�HG��z�A�;*

A2S/average_reward_1R�xB:��+       ��K	7L��z�A�;*

A2S/average_reward_1�xBD��+       ��K	��P��z�A�;*

A2S/average_reward_1)\xB=�0+       ��K	�l��z�A�<*

A2S/average_reward_1�G{B����+       ��K	�1q��z�A�<*

A2S/average_reward_1)\{B�RA�+       ��K	�~w��z�A�<*

A2S/average_reward_1\�{BG���+       ��K	௷��z�A�>*

A2S/average_reward_1���B|�+       ��K	m����z�A�>*

A2S/average_reward_1�Q�BH�t+       ��K	�'��z�A�@*

A2S/average_reward_1.�B��9�+       ��K	w2��z�A�@*

A2S/average_reward_1.�B��'+       ��K	o6��z�A�@*

A2S/average_reward_1 ��B�[f$+       ��K	pQ��z�A�B*

A2S/average_reward_1f�B�C,+       ��K	�u_��z�A�B*

A2S/average_reward_1�mB9EJ�+       ��K	��g��z�A�B*

A2S/average_reward_1
�mB���+       ��K	�k���z�A�D*

A2S/average_reward_1�uB�]�+       ��K	vO���z�A�D*

A2S/average_reward_1=
uBr�Ci+       ��K	1�@��z�A�H*

A2S/average_reward_133�B�9+       ��K	��C��z�A�H*

A2S/average_reward_1
ׂB�tip+       ��K	˙h��z�A�I*

A2S/average_reward_1���BW��+       ��K	�p��z�A�I*

A2S/average_reward_1�ǄB,9R�+       ��K	ݵ��z�A�K*

A2S/average_reward_1���B|~�+       ��K	ȸ���z�A�K*

A2S/average_reward_1�Q�B��)+       ��K	P���z�A�L*

A2S/average_reward_1�L�B�#�(+       ��K	�A���z�A�L*

A2S/average_reward_1q=�B��(+       ��K	�Y���z�A�M*

A2S/average_reward_1�Bo���       L	vM	ZE��z�A�M*�

A2S/kl~+?

A2S/average_advantage�92�

A2S/policy_network_loss祀�

A2S/value_network_loss>ZC

A2S/q_network_loss�C���+       ��K	K��z�A�M*

A2S/average_reward_1R8�B���+       ��K	�T��z�A�M*

A2S/average_reward_1q��B� �+       ��K	�a��z�A�N*

A2S/average_reward_1��B����+       ��K	�p���z�A�O*

A2S/average_reward_1�ǐB4D��+       ��K	L����z�A�O*

A2S/average_reward_1���B喡�+       ��K	#	���z�A�P*

A2S/average_reward_1�BV���+       ��K	����z�A�P*

A2S/average_reward_1ף�B��,�+       ��K	�����z�A�Q*

A2S/average_reward_1�p�B�A�+       ��K	T���z�A�Q*

A2S/average_reward_1ff�B�=(�+       ��K	�����z�A�Q*

A2S/average_reward_1�k�B.��+       ��K	�G���z�A�R*

A2S/average_reward_133�B�%��+       ��K	�m���z�A�R*

A2S/average_reward_1)\�Bz���+       ��K	$���z�A�S*

A2S/average_reward_1q=�B쪵&+       ��K	����z�A�S*

A2S/average_reward_1���B�(IU+       ��K	{dD��z�A�T*

A2S/average_reward_1�B�(�+       ��K	NT��z�A�T*

A2S/average_reward_1�L�B���+       ��K	0�Z��z�A�U*

A2S/average_reward_1�k�Br+~*+       ��K	(Kn��z�A�U*

A2S/average_reward_1)\�B�
�H+       ��K	�{��z�A�U*

A2S/average_reward_1���BN�k+       ��K	'���z�A�V*

A2S/average_reward_133�B��ô+       ��K	���z�A�V*

A2S/average_reward_1�ǚB� >+       ��K	
ՠ��z�A�V*

A2S/average_reward_1��B��	+       ��K	�{���z�A�X*

A2S/average_reward_1��B�Hr�+       ��K	����z�A�X*

A2S/average_reward_1��B���Z+       ��K	�����z�A�X*

A2S/average_reward_1R��BP�-�+       ��K	��0��z�A�Z*

A2S/average_reward_1q��B	h�+       ��K	\g@��z�A�Z*

A2S/average_reward_1�Q�B��-R+       ��K	Z+O��z�A�[*

A2S/average_reward_1��B�p��+       ��K	ʸx��z�A�\*

A2S/average_reward_1�z�B`�)V+       ��K	�����z�A�\*

A2S/average_reward_1�p�BWK+       ��K	C����z�A�\*

A2S/average_reward_1�G�BmzE�+       ��K	�����z�A�]*

A2S/average_reward_1=
�B�+       ��K	����z�A�^*

A2S/average_reward_1ff�B�,�2+       ��K	(����z�A�_*

A2S/average_reward_1R8�B��6>+       ��K	
D���z�A�_*

A2S/average_reward_1 ��B��d+       ��K	����z�A�_*

A2S/average_reward_1���B�,:D+       ��K	����z�A�`*

A2S/average_reward_1H�B䈦+       ��K	K�.��z�A�`*

A2S/average_reward_1q��B	���+       ��K	w�6��z�A�`*

A2S/average_reward_1)ܔB?Uq+       ��K	�F��z�A�a*

A2S/average_reward_13��B��V��       L	vM	����z�A�a*�

A2S/kl�9?

A2S/average_advantage��?

A2S/policy_network_loss!@

A2S/value_network_loss�kkC

A2S/q_network_loss��eCI[�+       ��K	�����z�A�b*

A2S/average_reward_1.�Bu7 �+       ��K	����z�A�b*

A2S/average_reward_1)ܓBy��+       ��K	�����z�A�b*

A2S/average_reward_1�k�B�t��+       ��K	��z�A�d*

A2S/average_reward_1.�BX��+       ��K	>�&��z�A�e*

A2S/average_reward_1{��Bm3�+       ��K	��G��z�A�f*

A2S/average_reward_1
טBA`�+       ��K	@�L��z�A�f*

A2S/average_reward_1�B?�g�+       ��K	!vb��z�A�f*

A2S/average_reward_1��B�Y�+       ��K	7�h��z�A�f*

A2S/average_reward_13��B�@�?+       ��K	��y��z�A�g*

A2S/average_reward_1��BF+       ��K	���z�A�h*

A2S/average_reward_1f�B��+       ��K	����z�A�i*

A2S/average_reward_1=
�B���+       ��K	�����z�A�j*

A2S/average_reward_1R��B��M+       ��K	mD���z�A�j*

A2S/average_reward_1���B�
W�+       ��K	Y��z�A�k*

A2S/average_reward_1�ǟB�|\+       ��K	4���z�A�k*

A2S/average_reward_1���BoO�+       ��K	��@��z�A�l*

A2S/average_reward_1f�B�^��+       ��K	�rn��z�A�m*

A2S/average_reward_1�Q�B�nv(+       ��K	u)x��z�A�n*

A2S/average_reward_1\�B�&��+       ��K	�H���z�A�o*

A2S/average_reward_1)\�B���+       ��K	�q���z�A�o*

A2S/average_reward_1�k�B�E��+       ��K	����z�A�o*

A2S/average_reward_1=��B��P5+       ��K	6����z�A�p*

A2S/average_reward_1R��B�+       ��K	n����z�A�p*

A2S/average_reward_1���Bp�3�+       ��K	]����z�A�p*

A2S/average_reward_1�Bj/W+       ��K	�����z�A�p*

A2S/average_reward_1Ha�B�$ +       ��K	����z�A�q*

A2S/average_reward_1�G�B��f+       ��K	 ���z�A�q*

A2S/average_reward_133�B�y��+       ��K	z$��z�A�r*

A2S/average_reward_1R��B�]�+       ��K	� 1��z�A�r*

A2S/average_reward_1��BwC@+       ��K	�T��z�A�s*

A2S/average_reward_1�z�By��+       ��K	��[��z�A�s*

A2S/average_reward_1�B�B��	�+       ��K	�e��z�A�s*

A2S/average_reward_1�k�BJd}+       ��K	�Wo��z�A�t*

A2S/average_reward_1H�B���	+       ��K	,Yv��z�A�t*

A2S/average_reward_1�u�B���+       ��K	����z�A�u*

A2S/average_reward_1{�B����+       ��K	E����z�A�v*

A2S/average_reward_1=
�B��+       ��K	����z�A�x*

A2S/average_reward_1��B3�<+       ��K	����z�A�x*

A2S/average_reward_1\��B}��O+       ��K	�&��z�A�x*

A2S/average_reward_1R��B�<&ߖ       L	vM	"�p��z�A�x*�

A2S/kldO8?

A2S/average_advantagemi�?

A2S/policy_network_lossbE@

A2S/value_network_loss���C

A2S/q_network_lossnފC4 �+       ��K	�a���z�A�z*

A2S/average_reward_1�(�B5�M1+       ��K	iF���z�A�}*

A2S/average_reward_1ף�B���+       ��K	���z�A�}*

A2S/average_reward_1\�B|���+       ��K	��J��z�A�*

A2S/average_reward_1��B�a��,       ���E	Ap��z�AÀ*

A2S/average_reward_1{��B�~,       ���E	����z�A��*

A2S/average_reward_1�B�B��
,       ���E	ˆ��z�A��*

A2S/average_reward_1��BLzH,       ���E	����z�AÃ*

A2S/average_reward_1���BK���,       ���E	�����z�A��*

A2S/average_reward_1��B<ʈ ,       ���E	~)��z�A��*

A2S/average_reward_1.�B�,!,       ���E	f0M��z�A��*

A2S/average_reward_1 ��B�Q�,       ���E	]����z�A�*

A2S/average_reward_1�ªB���,       ���E	�����z�A��*

A2S/average_reward_1�u�B�P��,       ���E	���z�A�*

A2S/average_reward_1�ǪB�3�#,       ���E	�� ��z�A��*

A2S/average_reward_1�G�B0�4�,       ���E	��3��z�A�*

A2S/average_reward_133�B�)��,       ���E	�Y��z�Aۍ*

A2S/average_reward_1ff�B��w�,       ���E	�Vw��z�AÎ*

A2S/average_reward_1��B�4�,       ���E	 ��z�A��*

A2S/average_reward_1���Be�@-,       ���E	E.%��z�Aԓ*

A2S/average_reward_1f�B�hs�,       ���E	B�;��z�A��*

A2S/average_reward_1f�BI�c�,       ���E	j'N��z�Aٔ*

A2S/average_reward_1�L�B�1=�,       ���E	c�]��z�A��*

A2S/average_reward_1�k�B���H,       ���E	4���z�A̗*

A2S/average_reward_1��B�	��,       ���E	�h���z�A�*

A2S/average_reward_1�ѺB/!�e,       ���E	�T���z�A��*

A2S/average_reward_1�p�B-��,       ���E	����z�A�*

A2S/average_reward_1R��B��C,       ���E	4�6��z�A��*

A2S/average_reward_1�k�B�PJ,       ���E	�����z�A��*

A2S/average_reward_1)\�Bs��t,       ���E	�-���z�A՞*

A2S/average_reward_1=
�B �ݿ,       ���E	F����z�A��*

A2S/average_reward_1�L�BM�R,       ���E	*����z�A��*

A2S/average_reward_1{�B��,       ���E	^���z�A��*

A2S/average_reward_1�k�B�ĥ,       ���E	.���z�A��*

A2S/average_reward_133�B<���,       ���E	��(��z�Aס*

A2S/average_reward_1���B���,       ���E	�'9��z�A��*

A2S/average_reward_1q��By:�9,       ���E	�9n��z�A��*

A2S/average_reward_1���B��5�,       ���E	=���z�A��*

A2S/average_reward_1���Bg�b,       ���E	�R���z�A¤*

A2S/average_reward_1Ha�B����,       ���E	�[ �z�A��*

A2S/average_reward_1Ha�BsFL"�       �v@�	��_ �z�A��*�

A2S/kl��?

A2S/average_advantage[���

A2S/policy_network_lossjZ�>

A2S/value_network_loss%��C

A2S/q_network_loss#��Cz�,       ���E	��d �z�A��*

A2S/average_reward_13��B}��*,       ���E	�^.�z�A�*

A2S/average_reward_1���B��P�,       ���E	�%o�z�Aî*

A2S/average_reward_1R��B�ޑ�,       ���E	uH��z�A��*

A2S/average_reward_1���B���K,       ���E	�,<�z�A۳*

A2S/average_reward_1
W�B`�+�,       ���E	J %�z�A��*

A2S/average_reward_1��B�D��,       ���E	;�3�z�A��*

A2S/average_reward_1�ClH�,       ���E	|�_�z�A��*

A2S/average_reward_1��C~�N,       ���E	��Q�z�A��*

A2S/average_reward_1��	CW�F�,       ���E	�3��z�A��*

A2S/average_reward_1R�C��D,       ���E	2_g�z�A��*

A2S/average_reward_1RxC[e�,       ���E	���z�A��*

A2S/average_reward_1��C�?��,       ���E	��C�z�A��*

A2S/average_reward_1�C�M,       ���E	V�n�z�A��*

A2S/average_reward_1\CY��,       ���E	9{��z�A��*

A2S/average_reward_1q�C'��_,       ���E	�B$�z�A��*

A2S/average_reward_1)�C�of,       ���E	<uZ�z�A��*

A2S/average_reward_1=J C��^�,       ���E	4q��z�A��*

A2S/average_reward_1
#C�x��,       ���E	����z�A��*

A2S/average_reward_1 �$C���,       ���E	kb�	�z�A��*

A2S/average_reward_1�)C�i?�,       ���E	���	�z�A��*

A2S/average_reward_1�k*C��`�,       ���E		��	�z�A��*

A2S/average_reward_1�+CV)�,       ���E	�K+
�z�A��*

A2S/average_reward_1 �,C�<��,       ���E	ϴ�
�z�A��*

A2S/average_reward_1=�0C���~,       ���E	����z�A��*

A2S/average_reward_1��7C��!�,       ���E	 �e�z�A��*

A2S/average_reward_1�<C�`s,       ���E	��p�z�A��*

A2S/average_reward_1��<C�t,       ���E	�˯�z�A��*

A2S/average_reward_1
>C2�5�,       ���E	���z�A��*

A2S/average_reward_1H�GC=���,       ���E	��j�z�Aц*

A2S/average_reward_1��KC.N�!,       ���E	1��z�A��*

A2S/average_reward_1)�SCu5�/,       ���E	����z�A�*

A2S/average_reward_1�:VC��U,       ���E	C���z�A��*

A2S/average_reward_1 @UC��o,       ���E	F��z�A��*

A2S/average_reward_1RxZCƩEB,       ���E	�ڌ�z�A֕*

A2S/average_reward_1{�YC_,       ���E	ߔT�z�A�*

A2S/average_reward_1f�_CdCC,       ���E	��e�z�AԢ*

A2S/average_reward_1
�hC�@%,       ���E	�%s�z�A��*

A2S/average_reward_1��gC~`��,       ���E	@6o�z�A��*

A2S/average_reward_1H�pCW�0,       ���E	��{�z�A�*

A2S/average_reward_1��oC�{h�       �v@�	|)��z�A�*�

A2S/kl��?

A2S/average_advantageC��

A2S/policy_network_lossy�C?

A2S/value_network_lossJ3�C

A2S/q_network_loss���C5�{,,       ���E	d#�z�A��*

A2S/average_reward_1��qCG(�,       ���E	�*�z�A��*

A2S/average_reward_1\�{C���P,       ���E	�y��z�AǷ*

A2S/average_reward_1H!C����,       ���E	���z�A��*

A2S/average_reward_13s�C�bY�,       ���E	���z�A��*

A2S/average_reward_1׃�C&W��,       ���E	p���z�A��*

A2S/average_reward_1�5�C]���,       ���E	�/!�z�A��*

A2S/average_reward_1
בC���,       ���E	t"6�z�A��*

A2S/average_reward_1���CW`��,       ���E	��<�z�A��*

A2S/average_reward_1�ޚCӝE�,       ���E	A=G�z�A��*

A2S/average_reward_1%�C��-�,       ���E	�z�A��*

A2S/average_reward_1Õ�C	ʁ],       ���E	J�@�z�A��*

A2S/average_reward_1�"�C1b��,       ���E	���z�A��*

A2S/average_reward_1���C�]�,       ���E	��5 �z�A��*

A2S/average_reward_1���C�Oq,       ���E	�Z!�z�A��*

A2S/average_reward_1�g�C�3 ,       ���E	�og"�z�AՏ*

A2S/average_reward_1Rx�C_'.[,       ���E	�J~#�z�A��*

A2S/average_reward_1=*�CyP?,       ���E	_y$�z�A��*

A2S/average_reward_1E�C�Q�,       ���E	�!�%�z�A��*

A2S/average_reward_1��C
���,       ���E	ر�&�z�A�*

A2S/average_reward_1
��C)�;,       ���E	T
�'�z�Aٵ*

A2S/average_reward_1q��C��,       ���E	�֬(�z�A�*

A2S/average_reward_1���C��d�,       ���E	�$�)�z�A��*

A2S/average_reward_1=��CT�,       ���E		�*�z�A��*

A2S/average_reward_1��C��",       ���E	�f�+�z�A��*

A2S/average_reward_1q�C�],",       ���E	:�,�z�A��*

A2S/average_reward_1���C��Y�,       ���E	6.�z�A��*

A2S/average_reward_1���C��T,       ���E	�a�.�z�A��*

A2S/average_reward_1���C%|3�,       ���E	|�/�z�A��*

A2S/average_reward_1\��C��,       ���E	{��0�z�A��*

A2S/average_reward_1���C5ۨ{,       ���E	��2�z�A��*

A2S/average_reward_1)��C����,       ���E	�ok2�z�A��*

A2S/average_reward_1�p�C�غ,       ���E	4��3�z�A��*

A2S/average_reward_1�9�C��,       ���E	ī�4�z�A��*

A2S/average_reward_1���C=m�,       ���E	�@�5�z�A�*

A2S/average_reward_1���C��c,       ���E	���6�z�A��*

A2S/average_reward_1� D=�,       ���E	��6�z�A��*

A2S/average_reward_1���CN��,       ���E	��8�z�A��*

A2S/average_reward_1=D1<�,       ���E	�489�z�AѬ*

A2S/average_reward_1=
DͰf6,       ���E	��j:�z�A��*

A2S/average_reward_1\_D�O;Z�       �v@�	� �:�z�A��*�

A2S/kl��>

A2S/average_advantage��?

A2S/policy_network_loss�p>

A2S/value_network_loss��XC

A2S/q_network_loss�UCď6[,       ���E	
�;�z�A��*

A2S/average_reward_1q�DSE2,       ���E	�><�z�A��*

A2S/average_reward_1�(Ddo0h,       ���E	U[`=�z�A��*

A2S/average_reward_1
�
D4B|