       ЃK"	  РПzжAbrain.Event:2§Rж;     ўЇ_Ё	iЖРПzжA"ї
s
A2S/observationsPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
A2S/advantagesPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
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
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
W
A2S/average_rewardPlaceholder*
shape:*
dtype0*
_output_shapes
:
ё
VA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB"      
у
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
у
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
р
^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
seed2*
dtype0*
_output_shapes

:
ђ
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes
: 

TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
і
PA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
ѓ
5A2S/backup_policy_network/backup_policy_network/fc0/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
ы
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
№
:A2S/backup_policy_network/backup_policy_network/fc0/w/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
о
GA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ы
5A2S/backup_policy_network/backup_policy_network/fc0/b
VariableV2*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
о
<A2S/backup_policy_network/backup_policy_network/fc0/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/bGA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zeros*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ь
:A2S/backup_policy_network/backup_policy_network/fc0/b/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
_output_shapes
:
а
 A2S/backup_policy_network/MatMulMatMulA2S/observations:A2S/backup_policy_network/backup_policy_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Д
A2S/backup_policy_network/addAdd A2S/backup_policy_network/MatMul:A2S/backup_policy_network/backup_policy_network/fc0/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
Ф
:A2S/backup_policy_network/LayerNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
б
(A2S/backup_policy_network/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
	container 
Њ
/A2S/backup_policy_network/LayerNorm/beta/AssignAssign(A2S/backup_policy_network/LayerNorm/beta:A2S/backup_policy_network/LayerNorm/beta/Initializer/zeros*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
Х
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta
Х
:A2S/backup_policy_network/LayerNorm/gamma/Initializer/onesConst*
_output_shapes
:*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
valueB*  ?*
dtype0
г
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
­
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Ш
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:

BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ъ
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
Ќ
8A2S/backup_policy_network/LayerNorm/moments/StopGradientStopGradient0A2S/backup_policy_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
н
=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_policy_network/add8A2S/backup_policy_network/LayerNorm/moments/StopGradient*'
_output_shapes
:џџџџџџџџџ*
T0

FA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
x
3A2S/backup_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
е
1A2S/backup_policy_network/LayerNorm/batchnorm/addAdd4A2S/backup_policy_network/LayerNorm/moments/variance3A2S/backup_policy_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
3A2S/backup_policy_network/LayerNorm/batchnorm/RsqrtRsqrt1A2S/backup_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Я
1A2S/backup_policy_network/LayerNorm/batchnorm/mulMul3A2S/backup_policy_network/LayerNorm/batchnorm/Rsqrt.A2S/backup_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
О
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_1MulA2S/backup_policy_network/add1A2S/backup_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
б
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2Mul0A2S/backup_policy_network/LayerNorm/moments/mean1A2S/backup_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
1A2S/backup_policy_network/LayerNorm/batchnorm/subSub-A2S/backup_policy_network/LayerNorm/beta/read3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
д
3A2S/backup_policy_network/LayerNorm/batchnorm/add_1Add3A2S/backup_policy_network/LayerNorm/batchnorm/mul_11A2S/backup_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
d
A2S/backup_policy_network/mul/xConst*
_output_shapes
: *
valueB
 *?*
dtype0
Ќ
A2S/backup_policy_network/mulMulA2S/backup_policy_network/mul/x3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/backup_policy_network/AbsAbs3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
f
!A2S/backup_policy_network/mul_1/xConst*
_output_shapes
: *
valueB
 *ЭЬЬ>*
dtype0

A2S/backup_policy_network/mul_1Mul!A2S/backup_policy_network/mul_1/xA2S/backup_policy_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/backup_policy_network/add_1AddA2S/backup_policy_network/mulA2S/backup_policy_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
p
+A2S/backup_policy_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ё
VA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
у
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
у
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *ЭЬЬ=*
dtype0
р
^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
seed28*
dtype0*
_output_shapes

:*

seed
ђ
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes
: 

TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
і
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
ѓ
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
ы
<A2S/backup_policy_network/backup_policy_network/out/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/wPA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
validate_shape(
№
:A2S/backup_policy_network/backup_policy_network/out/w/readIdentity5A2S/backup_policy_network/backup_policy_network/out/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
о
GA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ы
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
о
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ь
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:
с
"A2S/backup_policy_network/MatMul_1MatMulA2S/backup_policy_network/add_1:A2S/backup_policy_network/backup_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
И
A2S/backup_policy_network/add_2Add"A2S/backup_policy_network/MatMul_1:A2S/backup_policy_network/backup_policy_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
щ
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
л
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
л
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
д
ZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shape*
seed2H*
dtype0*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
т
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes
: 
є
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
ц
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
ы
1A2S/best_policy_network/best_policy_network/fc0/w
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
л
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
ф
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
ж
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
у
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
Ю
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(
р
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
Ъ
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ќ
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
Р
8A2S/best_policy_network/LayerNorm/beta/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Э
&A2S/best_policy_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape:
Ђ
-A2S/best_policy_network/LayerNorm/beta/AssignAssign&A2S/best_policy_network/LayerNorm/beta8A2S/best_policy_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
П
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
С
8A2S/best_policy_network/LayerNorm/gamma/Initializer/onesConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Я
'A2S/best_policy_network/LayerNorm/gamma
VariableV2*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ѕ
.A2S/best_policy_network/LayerNorm/gamma/AssignAssign'A2S/best_policy_network/LayerNorm/gamma8A2S/best_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Т
,A2S/best_policy_network/LayerNorm/gamma/readIdentity'A2S/best_policy_network/LayerNorm/gamma*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:

@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ф
.A2S/best_policy_network/LayerNorm/moments/meanMeanA2S/best_policy_network/add@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
Ј
6A2S/best_policy_network/LayerNorm/moments/StopGradientStopGradient.A2S/best_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:џџџџџџџџџ
з
;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

DA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
v
1A2S/best_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
Я
/A2S/best_policy_network/LayerNorm/batchnorm/addAdd2A2S/best_policy_network/LayerNorm/moments/variance1A2S/best_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:џџџџџџџџџ

1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtRsqrt/A2S/best_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
/A2S/best_policy_network/LayerNorm/batchnorm/mulMul1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt,A2S/best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
И
1A2S/best_policy_network/LayerNorm/batchnorm/mul_1MulA2S/best_policy_network/add/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
1A2S/best_policy_network/LayerNorm/batchnorm/mul_2Mul.A2S/best_policy_network/LayerNorm/moments/mean/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ш
/A2S/best_policy_network/LayerNorm/batchnorm/subSub+A2S/best_policy_network/LayerNorm/beta/read1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
1A2S/best_policy_network/LayerNorm/batchnorm/add_1Add1A2S/best_policy_network/LayerNorm/batchnorm/mul_1/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
b
A2S/best_policy_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
І
A2S/best_policy_network/mulMulA2S/best_policy_network/mul/x1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_policy_network/AbsAbs1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
d
A2S/best_policy_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ>

A2S/best_policy_network/mul_1MulA2S/best_policy_network/mul_1/xA2S/best_policy_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_policy_network/add_1AddA2S/best_policy_network/mulA2S/best_policy_network/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
n
)A2S/best_policy_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
щ
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
л
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *ЭЬЬН
л
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *ЭЬЬ=*
dtype0
д
ZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2u
т
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
є
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
ц
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
ы
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
л
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
ф
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
ж
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    
у
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
Ю
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
р
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
й
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/add_16A2S/best_policy_network/best_policy_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
А
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
э
TA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB"      *
dtype0
п
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  П*
dtype0
п
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  ?
л
\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
seed2
ъ
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes
: 
ќ
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
ю
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
я
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
у
:A2S/backup_value_network/backup_value_network/fc0/w/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/wNA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
ъ
8A2S/backup_value_network/backup_value_network/fc0/w/readIdentity3A2S/backup_value_network/backup_value_network/fc0/w*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
к
EA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zerosConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ч
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
ж
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(*
_output_shapes
:
ц
8A2S/backup_value_network/backup_value_network/fc0/b/readIdentity3A2S/backup_value_network/backup_value_network/fc0/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
_output_shapes
:
Э
A2S/backup_value_network/MatMulMatMulA2S/observations8A2S/backup_value_network/backup_value_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
А
A2S/backup_value_network/addAddA2S/backup_value_network/MatMul8A2S/backup_value_network/backup_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
Т
9A2S/backup_value_network/LayerNorm/beta/Initializer/zerosConst*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Я
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
І
.A2S/backup_value_network/LayerNorm/beta/AssignAssign'A2S/backup_value_network/LayerNorm/beta9A2S/backup_value_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
Т
,A2S/backup_value_network/LayerNorm/beta/readIdentity'A2S/backup_value_network/LayerNorm/beta*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
_output_shapes
:
У
9A2S/backup_value_network/LayerNorm/gamma/Initializer/onesConst*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
б
(A2S/backup_value_network/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
	container 
Љ
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Х
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:

AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ч
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0
Њ
7A2S/backup_value_network/LayerNorm/moments/StopGradientStopGradient/A2S/backup_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:џџџџџџџџџ
к
<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_value_network/add7A2S/backup_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

EA2S/backup_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
w
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
в
0A2S/backup_value_network/LayerNorm/batchnorm/addAdd3A2S/backup_value_network/LayerNorm/moments/variance2A2S/backup_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

2A2S/backup_value_network/LayerNorm/batchnorm/RsqrtRsqrt0A2S/backup_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
0A2S/backup_value_network/LayerNorm/batchnorm/mulMul2A2S/backup_value_network/LayerNorm/batchnorm/Rsqrt-A2S/backup_value_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
Л
2A2S/backup_value_network/LayerNorm/batchnorm/mul_1MulA2S/backup_value_network/add0A2S/backup_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
2A2S/backup_value_network/LayerNorm/batchnorm/mul_2Mul/A2S/backup_value_network/LayerNorm/moments/mean0A2S/backup_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
0A2S/backup_value_network/LayerNorm/batchnorm/subSub,A2S/backup_value_network/LayerNorm/beta/read2A2S/backup_value_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
б
2A2S/backup_value_network/LayerNorm/batchnorm/add_1Add2A2S/backup_value_network/LayerNorm/batchnorm/mul_10A2S/backup_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
c
A2S/backup_value_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
Љ
A2S/backup_value_network/mulMulA2S/backup_value_network/mul/x2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/backup_value_network/AbsAbs2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
e
 A2S/backup_value_network/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ>

A2S/backup_value_network/mul_1Mul A2S/backup_value_network/mul_1/xA2S/backup_value_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/backup_value_network/add_1AddA2S/backup_value_network/mulA2S/backup_value_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
o
*A2S/backup_value_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
э
TA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB"      
п
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
п
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
л
\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shape*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2В*
dtype0*
_output_shapes

:
ъ
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: 
ќ
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
ю
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
я
3A2S/backup_value_network/backup_value_network/out/w
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
	container 
у
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
ъ
8A2S/backup_value_network/backup_value_network/out/w/readIdentity3A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
к
EA2S/backup_value_network/backup_value_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
valueB*    
ч
3A2S/backup_value_network/backup_value_network/out/b
VariableV2*
_output_shapes
:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
	container *
shape:*
dtype0
ж
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
ц
8A2S/backup_value_network/backup_value_network/out/b/readIdentity3A2S/backup_value_network/backup_value_network/out/b*
_output_shapes
:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b
н
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Д
A2S/backup_value_network/add_2Add!A2S/backup_value_network/MatMul_18A2S/backup_value_network/backup_value_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
х
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"      *
dtype0
з
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
з
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Я
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2Т*
dtype0*
_output_shapes

:*

seed*
T0
к
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes
: 
ь
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
о
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
ч
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
г
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
о
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
в
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    
п
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
Ц
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
к
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
Ч
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ј
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
О
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Ы
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

,A2S/best_value_network/LayerNorm/beta/AssignAssign%A2S/best_value_network/LayerNorm/beta7A2S/best_value_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
М
*A2S/best_value_network/LayerNorm/beta/readIdentity%A2S/best_value_network/LayerNorm/beta*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
П
7A2S/best_value_network/LayerNorm/gamma/Initializer/onesConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Э
&A2S/best_value_network/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
	container 
Ё
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
П
+A2S/best_value_network/LayerNorm/gamma/readIdentity&A2S/best_value_network/LayerNorm/gamma*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:

?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
с
-A2S/best_value_network/LayerNorm/moments/meanMeanA2S/best_value_network/add?A2S/best_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
І
5A2S/best_value_network/LayerNorm/moments/StopGradientStopGradient-A2S/best_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:џџџџџџџџџ
д
:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

CA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

1A2S/best_value_network/LayerNorm/moments/varianceMean:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceCA2S/best_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0
u
0A2S/best_value_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *ЬМ+
Ь
.A2S/best_value_network/LayerNorm/batchnorm/addAdd1A2S/best_value_network/LayerNorm/moments/variance0A2S/best_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

0A2S/best_value_network/LayerNorm/batchnorm/RsqrtRsqrt.A2S/best_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
.A2S/best_value_network/LayerNorm/batchnorm/mulMul0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt+A2S/best_value_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
Е
0A2S/best_value_network/LayerNorm/batchnorm/mul_1MulA2S/best_value_network/add.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ш
0A2S/best_value_network/LayerNorm/batchnorm/mul_2Mul-A2S/best_value_network/LayerNorm/moments/mean.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Х
.A2S/best_value_network/LayerNorm/batchnorm/subSub*A2S/best_value_network/LayerNorm/beta/read0A2S/best_value_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
0A2S/best_value_network/LayerNorm/batchnorm/add_1Add0A2S/best_value_network/LayerNorm/batchnorm/mul_1.A2S/best_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
a
A2S/best_value_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
Ѓ
A2S/best_value_network/mulMulA2S/best_value_network/mul/x0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_value_network/AbsAbs0A2S/best_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
c
A2S/best_value_network/mul_1/xConst*
_output_shapes
: *
valueB
 *ЭЬЬ>*
dtype0

A2S/best_value_network/mul_1MulA2S/best_value_network/mul_1/xA2S/best_value_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_value_network/add_1AddA2S/best_value_network/mulA2S/best_value_network/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
m
(A2S/best_value_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
х
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
з
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
з
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Я
XA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2я
к
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes
: 
ь
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
о
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
ч
/A2S/best_value_network/best_value_network/out/w
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
г
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
о
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
в
AA2S/best_value_network/best_value_network/out/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
п
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
Ц
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
к
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
е
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ќ
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
b
A2S/Reshape/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

A2S/ReshapeReshapeA2S/backup_policy_network/add_2A2S/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
d
A2S/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

A2S/Reshape_1ReshapeA2S/best_policy_network/add_2A2S/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
A2S/Normal/locIdentityA2S/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
H
A2S/Normal/scaleIdentity	A2S/Const*
T0*
_output_shapes
: 
]
A2S/Normal_1/locIdentityA2S/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
L
A2S/Normal_1/scaleIdentityA2S/Const_1*
T0*
_output_shapes
: 
o
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
valueB
 *  ?*
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
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal_1/scale*
T0*
_output_shapes
: 
j
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal/scale*
T0*
_output_shapes
: 
Д
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0

(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal_1/locA2S/Normal/loc*
T0*'
_output_shapes
:џџџџџџџџџ

-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:џџџџџџџџџ
­
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 
Ф
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*
_output_shapes
: *
T0
~
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
_output_shapes
: *
T0
Ј
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
T0*
_output_shapes
: 
Ќ
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
T0*
_output_shapes
: 
Н
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:

A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
'A2S/Normal_2/batch_shape_tensor/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
Г
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
]
A2S/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
Q
A2S/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ

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
A2S/random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
А
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
seed2Ё*

seed*
T0

A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
h
A2S/addAddA2S/mulA2S/Normal_1/loc*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0
h
A2S/Reshape_2/shapeConst*!
valueB"џџџџ      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:џџџџџџџџџ
S
A2S/concat_1/axisConst*
_output_shapes
: *
value	B :*
dtype0

A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ*

Tidx0
н
LA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
Я
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  П
Я
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
У
TA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shape*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
seed2­*
dtype0*
_output_shapes

:*

seed*
T0
Ъ
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes
: 
м
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
Ю
FA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
п
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
У
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(
в
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
Ъ
=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
з
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
Ж
2A2S/backup_q_network/backup_q_network/fc0/b/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/b=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
validate_shape(*
_output_shapes
:
Ю
0A2S/backup_q_network/backup_q_network/fc0/b/readIdentity+A2S/backup_q_network/backup_q_network/fc0/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
_output_shapes
:
Н
A2S/backup_q_network/MatMulMatMulA2S/concat_10A2S/backup_q_network/backup_q_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
A2S/backup_q_network/addAddA2S/backup_q_network/MatMul0A2S/backup_q_network/backup_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
К
5A2S/backup_q_network/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
valueB*    
Ч
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

*A2S/backup_q_network/LayerNorm/beta/AssignAssign#A2S/backup_q_network/LayerNorm/beta5A2S/backup_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
Ж
(A2S/backup_q_network/LayerNorm/beta/readIdentity#A2S/backup_q_network/LayerNorm/beta*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
_output_shapes
:
Л
5A2S/backup_q_network/LayerNorm/gamma/Initializer/onesConst*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Щ
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

+A2S/backup_q_network/LayerNorm/gamma/AssignAssign$A2S/backup_q_network/LayerNorm/gamma5A2S/backup_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Й
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:

=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
л
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
Ђ
3A2S/backup_q_network/LayerNorm/moments/StopGradientStopGradient+A2S/backup_q_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_q_network/add3A2S/backup_q_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
s
.A2S/backup_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
Ц
,A2S/backup_q_network/LayerNorm/batchnorm/addAdd/A2S/backup_q_network/LayerNorm/moments/variance.A2S/backup_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

.A2S/backup_q_network/LayerNorm/batchnorm/RsqrtRsqrt,A2S/backup_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Р
,A2S/backup_q_network/LayerNorm/batchnorm/mulMul.A2S/backup_q_network/LayerNorm/batchnorm/Rsqrt)A2S/backup_q_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
Џ
.A2S/backup_q_network/LayerNorm/batchnorm/mul_1MulA2S/backup_q_network/add,A2S/backup_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Т
.A2S/backup_q_network/LayerNorm/batchnorm/mul_2Mul+A2S/backup_q_network/LayerNorm/moments/mean,A2S/backup_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
П
,A2S/backup_q_network/LayerNorm/batchnorm/subSub(A2S/backup_q_network/LayerNorm/beta/read.A2S/backup_q_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
Х
.A2S/backup_q_network/LayerNorm/batchnorm/add_1Add.A2S/backup_q_network/LayerNorm/batchnorm/mul_1,A2S/backup_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
_
A2S/backup_q_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 

A2S/backup_q_network/mulMulA2S/backup_q_network/mul/x.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/backup_q_network/AbsAbs.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
a
A2S/backup_q_network/mul_1/xConst*
_output_shapes
: *
valueB
 *ЭЬЬ>*
dtype0

A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/backup_q_network/add_1AddA2S/backup_q_network/mulA2S/backup_q_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
k
&A2S/backup_q_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
н
LA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB"      
Я
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/minConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
Я
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
У
TA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
seed2к
Ъ
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes
: 
м
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
Ю
FA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
п
+A2S/backup_q_network/backup_q_network/out/w
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
У
2A2S/backup_q_network/backup_q_network/out/w/AssignAssign+A2S/backup_q_network/backup_q_network/out/wFA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
validate_shape(*
_output_shapes

:
в
0A2S/backup_q_network/backup_q_network/out/w/readIdentity+A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
Ъ
=A2S/backup_q_network/backup_q_network/out/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
з
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
Ж
2A2S/backup_q_network/backup_q_network/out/b/AssignAssign+A2S/backup_q_network/backup_q_network/out/b=A2S/backup_q_network/backup_q_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b
Ю
0A2S/backup_q_network/backup_q_network/out/b/readIdentity+A2S/backup_q_network/backup_q_network/out/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
_output_shapes
:
Э
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Є
A2S/backup_q_network/add_2AddA2S/backup_q_network/MatMul_10A2S/backup_q_network/backup_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
е
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
Ч
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  П
Ч
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
З
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2ъ
К
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes
: 
Ь
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
О
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
з
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
Г
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
Ц
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
Т
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0
Я
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
І
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
Т
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
З
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
3A2S/best_q_network/LayerNorm/beta/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
У
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

(A2S/best_q_network/LayerNorm/beta/AssignAssign!A2S/best_q_network/LayerNorm/beta3A2S/best_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
А
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
З
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  ?
Х
"A2S/best_q_network/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma

)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
Г
'A2S/best_q_network/LayerNorm/gamma/readIdentity"A2S/best_q_network/LayerNorm/gamma*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:*
T0

;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
е
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0

1A2S/best_q_network/LayerNorm/moments/StopGradientStopGradient)A2S/best_q_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
Ш
6A2S/best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
§
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
q
,A2S/best_q_network/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *ЬМ+*
dtype0
Р
*A2S/best_q_network/LayerNorm/batchnorm/addAdd-A2S/best_q_network/LayerNorm/moments/variance,A2S/best_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

,A2S/best_q_network/LayerNorm/batchnorm/RsqrtRsqrt*A2S/best_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
К
*A2S/best_q_network/LayerNorm/batchnorm/mulMul,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
,A2S/best_q_network/LayerNorm/batchnorm/mul_1MulA2S/best_q_network/add*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
М
,A2S/best_q_network/LayerNorm/batchnorm/mul_2Mul)A2S/best_q_network/LayerNorm/moments/mean*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Й
*A2S/best_q_network/LayerNorm/batchnorm/subSub&A2S/best_q_network/LayerNorm/beta/read,A2S/best_q_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
П
,A2S/best_q_network/LayerNorm/batchnorm/add_1Add,A2S/best_q_network/LayerNorm/batchnorm/mul_1*A2S/best_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
]
A2S/best_q_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 

A2S/best_q_network/mulMulA2S/best_q_network/mul/x,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
}
A2S/best_q_network/AbsAbs,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
_
A2S/best_q_network/mul_1/xConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 

A2S/best_q_network/mul_1MulA2S/best_q_network/mul_1/xA2S/best_q_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_q_network/add_1AddA2S/best_q_network/mulA2S/best_q_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
i
$A2S/best_q_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
е
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
Ч
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *ЭЬЬН*
dtype0
Ч
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *ЭЬЬ=
З
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes

:*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
К
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes
: 
Ь
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
О
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
з
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
Г
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
Ц
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
Т
9A2S/best_q_network/best_q_network/out/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
Я
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
І
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
Т
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:*
T0
Х
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/add_1,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
}
%A2S/Normal_3/log_prob/standardize/subSubA2S/actionsA2S/Normal_1/loc*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
)A2S/Normal_3/log_prob/standardize/truedivRealDiv%A2S/Normal_3/log_prob/standardize/subA2S/Normal_1/scale*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/Normal_3/log_prob/SquareSquare)A2S/Normal_3/log_prob/standardize/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
`
A2S/Normal_3/log_prob/mul/xConst*
_output_shapes
: *
valueB
 *   П*
dtype0

A2S/Normal_3/log_prob/mulMulA2S/Normal_3/log_prob/mul/xA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:џџџџџџџџџ
U
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
T0*
_output_shapes
: 
`
A2S/Normal_3/log_prob/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 
y
A2S/Normal_3/log_prob/addAddA2S/Normal_3/log_prob/add/xA2S/Normal_3/log_prob/Log*
T0*
_output_shapes
: 

A2S/Normal_3/log_prob/subSubA2S/Normal_3/log_prob/mulA2S/Normal_3/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
[
A2S/NegNegA2S/Normal_3/log_prob/sub*
T0*'
_output_shapes
:џџџџџџџџџ
[
	A2S/mul_1MulA2S/NegA2S/advantages*
T0*'
_output_shapes
:џџџџџџџџџ
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
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
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
A2S/Mean_2*
_output_shapes
: *
T0

A2S/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add_2A2S/returns*'
_output_shapes
:џџџџџџџџџ*
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
: *
	keep_dims( *

Tidx0*
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

A2S/SquaredDifference_1SquaredDifferenceA2S/best_q_network/add_2A2S/returns*'
_output_shapes
:џџџџџџџџџ*
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
: *
	keep_dims( *

Tidx0*
T0
j
A2S/q_network_loss/tagsConst*
dtype0*
_output_shapes
: *#
valueB BA2S/q_network_loss
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
A2S/gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
Ј
%A2S/gradients/A2S/Mean_2_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_2_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
l
#A2S/gradients/A2S/Mean_2_grad/ShapeShape	A2S/mul_1*
_output_shapes
:*
T0*
out_type0
К
"A2S/gradients/A2S/Mean_2_grad/TileTile%A2S/gradients/A2S/Mean_2_grad/Reshape#A2S/gradients/A2S/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
Д
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
И
$A2S/gradients/A2S/Mean_2_grad/Prod_1Prod%A2S/gradients/A2S/Mean_2_grad/Shape_2%A2S/gradients/A2S/Mean_2_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
'A2S/gradients/A2S/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
 
%A2S/gradients/A2S/Mean_2_grad/MaximumMaximum$A2S/gradients/A2S/Mean_2_grad/Prod_1'A2S/gradients/A2S/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

&A2S/gradients/A2S/Mean_2_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_2_grad/Prod%A2S/gradients/A2S/Mean_2_grad/Maximum*
_output_shapes
: *
T0

"A2S/gradients/A2S/Mean_2_grad/CastCast&A2S/gradients/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Њ
%A2S/gradients/A2S/Mean_2_grad/truedivRealDiv"A2S/gradients/A2S/Mean_2_grad/Tile"A2S/gradients/A2S/Mean_2_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
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
в
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_2_grad/truedivA2S/advantages*
T0*'
_output_shapes
:џџџџџџџџџ
Н
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Е
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_2_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
У
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Л
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1

5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0

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

BA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
л
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
п
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1*
_output_shapes
:*
T0
и
6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
=A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
Т
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
З
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

4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1ShapeA2S/Normal_3/log_prob/Square*
T0*
out_type0*
_output_shapes
:

BA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:џџџџџџџџџ
э
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Я
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1MulA2S/Normal_3/log_prob/mul/xEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ы
6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Е
=A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
Б
EA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape*
_output_shapes
: 
Ш
GA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ф
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ю
3A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/x)A2S/Normal_3/log_prob/standardize/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
ь
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ї
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_3/log_prob/standardize/sub*
_output_shapes
:*
T0*
out_type0

DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
В
RA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ь
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1A2S/Normal_1/scale*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
 
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*'
_output_shapes
:џџџџџџџџџ*
T0
й
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*
T0*'
_output_shapes
:џџџџџџџџџ
п
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal_1/scale*'
_output_shapes
:џџџџџџџџџ*
T0
ј
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
х
MA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1

UA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape
ї
WA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 

>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
out_type0*
_output_shapes
:*
T0

@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal_1/loc*
T0*
out_type0*
_output_shapes
:
І
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Њ
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ў
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
І
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
_output_shapes
:*
T0

BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
й
IA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1
ђ
QA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ј
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1

&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
№
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

6A2S/gradients/A2S/best_policy_network/add_2_grad/ShapeShape A2S/best_policy_network/MatMul_1*
_output_shapes
:*
T0*
out_type0

8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ё
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
С
AA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1
в
IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ы
KA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1*
_output_shapes
:

:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
џ
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
Ш
DA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul=^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
м
LA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulE^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul
й
NA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1E^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:

6A2S/gradients/A2S/best_policy_network/add_1_grad/ShapeShapeA2S/best_policy_network/mul*
T0*
out_type0*
_output_shapes
:

8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1ShapeA2S/best_policy_network/mul_1*
T0*
out_type0*
_output_shapes
:

FA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ё
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
С
AA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
в
IA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
и
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
w
4A2S/gradients/A2S/best_policy_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ї
6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

DA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/mul_grad/Shape6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
щ
2A2S/gradients/A2S/best_policy_network/mul_grad/mulMulIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
ѓ
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
з
4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1MulA2S/best_policy_network/mul/xIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
љ
4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_1Sum4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1FA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ё
8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_16A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Л
?A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape9^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1
Й
GA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/mul_grad/Reshape@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape*
_output_shapes
: 
а
IA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
y
6A2S/gradients/A2S/best_policy_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1ShapeA2S/best_policy_network/Abs*
T0*
out_type0*
_output_shapes
:

FA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
з
4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulMulKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1A2S/best_policy_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0
љ
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
р
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
н
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
џ
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
С
AA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1
С
IA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape*
_output_shapes
: 
и
KA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
3A2S/gradients/A2S/best_policy_network/Abs_grad/SignSign1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
э
2A2S/gradients/A2S/best_policy_network/Abs_grad/mulMulKA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_13A2S/gradients/A2S/best_policy_network/Abs_grad/Sign*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Л
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/sub*
out_type0*
_output_shapes
:*
T0
Ъ
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
­
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Г
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
§
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
Ђ
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape
Ј
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
Л
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Ъ
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Е
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
­
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
џ
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_policy_network/add]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
§
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
Ђ
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape
Ј
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
Ф
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ш
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ь
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
К
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegNegHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
Ћ
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ї
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1

[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
 
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
И
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape.A2S/best_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
Л
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Ъ
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Е
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
­
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul.A2S/best_policy_network/LayerNorm/moments/mean]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Г
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
§
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
Ђ
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
ќ
A2S/gradients/AddN_1AddN_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ*
T0
Й
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ї
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtA2S/gradients/AddN_1*
T0*'
_output_shapes
:џџџџџџџџџ
Е
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
 
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
ї
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1

[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1

NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
К
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape2A2S/best_policy_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ф
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
З
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumSumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ї
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Л
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ї
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1

[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
Ц
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeShape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:

JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addAddDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modFloorModIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
ю
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeRangeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0

PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/FillFillMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
Ь
SA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ќ
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumSA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

NA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
Л
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeReshape[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencySA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
О
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileTileMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeNA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ш
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
П
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape2A2S/best_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ќ
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
А
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*
dtype0

OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1MaximumLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 

PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
д
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/CastCastPA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Ђ
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truedivRealDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
Ь
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
ш
dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
UA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Б
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulUA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
А
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradientN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Е
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
е
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ы
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
й
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
б
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
х
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

_A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpW^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeS^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ъ
gA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*i
_class_
][loc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ф
iA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*e
_class[
YWloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ђ
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:

FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addAdd@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modFloorModEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
_output_shapes
:*
T0

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
о
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeRangeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0

LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/FillFillIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0
И
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/MaximumMaximumOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
Е
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeReshape]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
В
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileTileIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Є
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_policy_network/add*
out_type0*
_output_shapes
:*
T0
З
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape.A2S/best_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:

GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
 
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Є
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*
dtype0

KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumHA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 

LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
Ь
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/CastCastLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truedivRealDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N

4A2S/gradients/A2S/best_policy_network/add_grad/ShapeShapeA2S/best_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:

6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
е
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
й
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ф
8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/add_grad/Sum_16A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Л
?A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/add_grad/Reshape9^A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1
Ъ
GA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/add_grad/Reshape@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
У
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
_output_shapes
:*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1

8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ю
:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Т
BA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul;^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1
д
JA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulC^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul
б
LA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1C^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:

A2S/beta1_power/initial_valueConst*
valueB
 *fff?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Ў
A2S/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
	container *
shape: 
е
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 

A2S/beta1_power/readIdentityA2S/beta1_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 

A2S/beta2_power/initial_valueConst*
valueB
 *wО?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Ў
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
е
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta

A2S/beta2_power/readIdentityA2S/beta2_power*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
ч
LA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
є
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
э
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/w/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(
і
?A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
щ
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
і
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1
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
ѓ
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
њ
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
п
LA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ь
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
щ
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/b/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ђ
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:*
T0
с
NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ю
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
я
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
і
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
Щ
AA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
ж
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
Н
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
б
4A2S/A2S/best_policy_network/LayerNorm/beta/Adam/readIdentity/A2S/A2S/best_policy_network/LayerNorm/beta/Adam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
Ы
CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
и
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
У
8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/AssignAssign1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
е
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/readIdentity1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:*
T0
Ы
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
и
0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam
VariableV2*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
С
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
д
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
Э
DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
к
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
Ч
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
и
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/readIdentity2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
ч
LA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
є
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
э
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/w/AdamLA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
і
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
щ
NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
і
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
ѓ
CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(
њ
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
п
LA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ь
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
щ
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
ђ
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
с
NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ю
<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
я
CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
і
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
 *wО?*
dtype0*
_output_shapes
: 
U
A2S/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
к
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/w:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:
г
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/b:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonIA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:
Ў
@A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam&A2S/best_policy_network/LayerNorm/beta/A2S/A2S/best_policy_network/LayerNorm/beta/Adam1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
Е
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
м
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/w:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
use_nesterov( *
_output_shapes

:
е
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/b:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonKA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
use_nesterov( *
_output_shapes
:
д
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
Н
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
ж
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
С
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
ѕ
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
 *  ?
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
Ў
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
Р
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
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
К
$A2S/gradients_1/A2S/Mean_3_grad/ProdProd'A2S/gradients_1/A2S/Mean_3_grad/Shape_1%A2S/gradients_1/A2S/Mean_3_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'A2S/gradients_1/A2S/Mean_3_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
О
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_1/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'A2S/gradients_1/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_3_grad/Prod_1)A2S/gradients_1/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
Є
(A2S/gradients_1/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_3_grad/Prod'A2S/gradients_1/A2S/Mean_3_grad/Maximum*
T0*
_output_shapes
: 

$A2S/gradients_1/A2S/Mean_3_grad/CastCast(A2S/gradients_1/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

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
ќ
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
 
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
У
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
М
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/best_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_3_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
щ
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
п
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
э
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

.A2S/gradients_1/A2S/SquaredDifference_grad/NegNeg4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
;A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_depsNoOp3^A2S/gradients_1/A2S/SquaredDifference_grad/Reshape/^A2S/gradients_1/A2S/SquaredDifference_grad/Neg
К
CA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIdentity2A2S/gradients_1/A2S/SquaredDifference_grad/Reshape<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*E
_class;
97loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Д
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ*
T0

7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

GA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
э
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ф
BA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
ж
JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape
Я
LA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1*
_output_shapes
:

;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulMatMulJA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1MatMulA2S/best_value_network/add_1JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Ы
EA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_depsNoOp<^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul>^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1
р
MA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIdentity;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulF^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
н
OA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1Identity=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1F^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*P
_classF
DBloc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:*
T0

7A2S/gradients_1/A2S/best_value_network/add_1_grad/ShapeShapeA2S/best_value_network/mul*
_output_shapes
:*
T0*
out_type0

9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1ShapeA2S/best_value_network/mul_1*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5A2S/gradients_1/A2S/best_value_network/add_1_grad/SumSumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_1SumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
њ
;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ф
BA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1
ж
JA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape
м
LA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
x
5A2S/gradients_1/A2S/best_value_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ї
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

EA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
3A2S/gradients_1/A2S/best_value_network/mul_grad/mulMulJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
і
3A2S/gradients_1/A2S/best_value_network/mul_grad/SumSum3A2S/gradients_1/A2S/best_value_network/mul_grad/mulEA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
н
7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/mul_grad/Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
и
5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1MulA2S/best_value_network/mul/xJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ќ
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
О
@A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1
Н
HA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape*
_output_shapes
: 
д
JA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
z
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1ShapeA2S/best_value_network/Abs*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
и
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulMulLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1A2S/best_value_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0
ќ
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
у
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
о
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1MulA2S/best_value_network/mul_1/xLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
њ
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
BA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1
Х
JA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape*
_output_shapes
: *
T0
м
LA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
4A2S/gradients_1/A2S/best_value_network/Abs_grad/SignSign0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
№
3A2S/gradients_1/A2S/best_value_network/Abs_grad/mulMulLA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_14A2S/gradients_1/A2S/best_value_network/Abs_grad/Sign*'
_output_shapes
:џџџџџџџџџ*
T0
І
A2S/gradients_1/AddNAddNJA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_13A2S/gradients_1/A2S/best_value_network/Abs_grad/mul*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Л
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Л
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
Э
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
І
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ќ
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ѕ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
Л
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Э
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
И
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
О
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
І
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ќ
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Л
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
_output_shapes
:*
T0*
out_type0
Ч
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ы
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
Tshape0*
_output_shapes
:*
T0
Я
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegNegIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
Ў
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
њ
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1

\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape
Є
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
И
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-A2S/best_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
Л
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Э
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
И
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
А
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-A2S/best_value_network/LayerNorm/moments/mean^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
О
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ж
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
І
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape
Ќ
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1

A2S/gradients_1/AddN_1AddN`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Й
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ч
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Х
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
В
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ь
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0A2S/best_value_network/LayerNorm/batchnorm/RsqrtA2S/gradients_1/AddN_1*
T0*'
_output_shapes
:џџџџџџџџџ
И
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѓ
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
њ
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1

\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:

OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
К
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeShape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
К
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
О
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
њ
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1

\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
Ц
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0

KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addAddCA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modFloorModJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0

RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ђ
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeRangeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*
_output_shapes
:

QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
б
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
T0*
N

PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Џ
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/MaximumMaximumTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0

OA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordivFloorDivLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
О
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
С
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0
Ш
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
П
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Џ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Г
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1MaximumMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 

QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
ж
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/CastCastQA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Ѕ
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truedivRealDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
Ь
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5A2S/best_value_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
ы
eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ь
VA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarConstO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Д
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulMulVA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
А
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradientO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
И
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
и
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ю
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
м
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
д
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ч
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegYA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

`A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpX^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeT^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ю
hA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshapea^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*j
_class`
^\loc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape
Ш
jA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitySA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Nega^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*f
_class\
ZXloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ђ
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_value_network/add*
out_type0*
_output_shapes
:*
T0

GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ќ
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addAdd?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modFloorModFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
т
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeRangeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0

MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/FillFillJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
Н
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Ѓ
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/MaximumMaximumPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
И
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeReshape^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Е
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

Tmultiples0
Є
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
out_type0*
_output_shapes
:*
T0
З
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3Shape-A2S/best_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:

HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Ѓ
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ї
IA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1ProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumIA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 

MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
Ю
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/CastCastMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truedivRealDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
б
A2S/gradients_1/AddN_2AddN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyhA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truediv*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

5A2S/gradients_1/A2S/best_value_network/add_grad/ShapeShapeA2S/best_value_network/MatMul*
T0*
out_type0*
_output_shapes
:

7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
й
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ю
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
н
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ч
9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_17A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
О
@A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1
Ю
HA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ч
JA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1*
_output_shapes
:

9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
№
;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Х
CA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul<^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1
и
KA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulD^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
е
MA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1D^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1*
_output_shapes

:

A2S/beta1_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
Џ
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
к
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 

A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 

A2S/beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
Џ
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
к
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
у
JA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
№
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
х
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
№
=A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:*
T0
х
LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    
ђ
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
ы
AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
є
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
л
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ш
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
с
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/b/AdamJA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(
ь
=A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
н
LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ъ
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
ч
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
№
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
Ч
@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
д
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
Й
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ю
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
Щ
BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    
ж
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
П
7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/AssignAssign0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
в
5A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/readIdentity0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
Щ
AA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
ж
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
Н
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/AssignAssign/A2S/A2S/best_value_network/LayerNorm/gamma/AdamAA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zeros*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
б
4A2S/A2S/best_value_network/LayerNorm/gamma/Adam/readIdentity/A2S/A2S/best_value_network/LayerNorm/gamma/Adam*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
Ы
CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
и
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
У
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
е
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
у
JA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
№
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
х
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
№
=A2S/A2S/best_value_network/best_value_network/out/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/w/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
х
LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    
ђ
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
ы
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
є
?A2S/A2S/best_value_network/best_value_network/out/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
л
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0
ш
8A2S/A2S/best_value_network/best_value_network/out/b/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
с
?A2S/A2S/best_value_network/best_value_network/out/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/b/AdamJA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
ь
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
н
LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ъ
:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ч
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(
№
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
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
 *wО?*
dtype0*
_output_shapes
: 
W
A2S/Adam_1/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wЬ+2
н
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
use_nesterov( *
_output_shapes

:
ж
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/b8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonJA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
use_nesterov( *
_output_shapes
:
Ж
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
Н
BA2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&A2S/best_value_network/LayerNorm/gamma/A2S/A2S/best_value_network/LayerNorm/gamma/Adam1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
п
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/w8A2S/A2S/best_value_network/best_value_network/out/w/Adam:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
use_nesterov( *
_output_shapes

:
и
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( *
_output_shapes
:
л
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
Т
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(
н
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
Ц
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
§

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
 *  ?*
dtype0
k
A2S/gradients_2/FillFillA2S/gradients_2/ShapeA2S/gradients_2/Const*
T0*
_output_shapes
: 
~
-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Ў
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
Р
$A2S/gradients_2/A2S/Mean_4_grad/TileTile'A2S/gradients_2/A2S/Mean_4_grad/Reshape%A2S/gradients_2/A2S/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
К
$A2S/gradients_2/A2S/Mean_4_grad/ProdProd'A2S/gradients_2/A2S/Mean_4_grad/Shape_1%A2S/gradients_2/A2S/Mean_4_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
'A2S/gradients_2/A2S/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
О
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
І
'A2S/gradients_2/A2S/Mean_4_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_4_grad/Prod_1)A2S/gradients_2/A2S/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 
Є
(A2S/gradients_2/A2S/Mean_4_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_4_grad/Prod'A2S/gradients_2/A2S/Mean_4_grad/Maximum*
T0*
_output_shapes
: 

$A2S/gradients_2/A2S/Mean_4_grad/CastCast(A2S/gradients_2/A2S/Mean_4_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_2/A2S/Mean_4_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_4_grad/Tile$A2S/gradients_2/A2S/Mean_4_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
T0*
out_type0*
_output_shapes
:

4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1ShapeA2S/returns*
out_type0*
_output_shapes
:*
T0

BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ђ
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ч
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_4_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
К
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/best_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Я
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
я
0A2S/gradients_2/A2S/SquaredDifference_1_grad/SumSum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ѓ
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ы
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ё
0A2S/gradients_2/A2S/SquaredDifference_1_grad/NegNeg6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
=A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_depsNoOp5^A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape1^A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg
Т
EA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyIdentity4A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg*'
_output_shapes
:џџџџџџџџџ

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

CA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
с
7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
И
>A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1
Ц
FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
П
HA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1

7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
є
9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1MatMulA2S/best_q_network/add_1FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
П
AA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_depsNoOp8^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul:^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1
а
IA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyIdentity7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulB^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul
Э
KA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1Identity9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1B^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1*
_output_shapes

:

3A2S/gradients_2/A2S/best_q_network/add_1_grad/ShapeShapeA2S/best_q_network/mul*
_output_shapes
:*
T0*
out_type0

5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1ShapeA2S/best_q_network/mul_1*
out_type0*
_output_shapes
:*
T0

CA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ш
5A2S/gradients_2/A2S/best_q_network/add_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_1SumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ю
7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
И
>A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1
Ц
FA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ь
HA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
t
1A2S/gradients_2/A2S/best_q_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
џ
AA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
о
/A2S/gradients_2/A2S/best_q_network/mul_grad/mulMulFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/A2S/gradients_2/A2S/best_q_network/mul_grad/SumSum/A2S/gradients_2/A2S/best_q_network/mul_grad/mulAA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
3A2S/gradients_2/A2S/best_q_network/mul_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/mul_grad/Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
Ь
1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1MulA2S/best_q_network/mul/xFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
№
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ш
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
В
<A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
­
DA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape
Ф
FA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
v
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1ShapeA2S/best_q_network/Abs*
T0*
out_type0*
_output_shapes
:

CA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulMulHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1A2S/best_q_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ
№
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
з
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
в
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
і
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ю
7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
И
>A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
Е
FA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape*
_output_shapes
: *
T0
Ь
HA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

0A2S/gradients_2/A2S/best_q_network/Abs_grad/SignSign,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
ф
/A2S/gradients_2/A2S/best_q_network/Abs_grad/mulMulHA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_10A2S/gradients_2/A2S/best_q_network/Abs_grad/Sign*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/gradients_2/AddNAddNFA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1/A2S/gradients_2/A2S/best_q_network/Abs_grad/mul*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Г
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
Г
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/sub*
out_type0*
_output_shapes
:*
T0
С
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ћ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Є
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
џ
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Њ
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
є
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:
Г
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
С
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Є
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
є
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
В
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Г
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
Л
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
П
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
У
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Д
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
Ђ
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ю
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1

XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
А
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
Г
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
С
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Є
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul)A2S/best_q_network/LayerNorm/moments/meanZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
В
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Њ
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
ѕ
A2S/gradients_2/AddN_1AddN\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Б
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
Л
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
І
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
ю
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1

XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:

KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad,A2S/best_q_network/LayerNorm/batchnorm/RsqrtXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
В
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Л
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
В
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1SumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ю
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1

XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1
О
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeShape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ќ
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addAdd?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modFloorModFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
т
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*
_output_shapes
:

MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :

GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/FillFillJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
Н
PA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
T0*
N

LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
Ѓ
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

KA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordivFloorDivHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum*
_output_shapes
:*
T0
В
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeReshapeXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Е
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0
Р
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2Shape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
З
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3Shape-A2S/best_q_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ѓ
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ї
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0

MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
Ю
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/CastCastMA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truedivRealDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
Ї
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
Ф
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
п
aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeSA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ф
RA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
Ј
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulMulRA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
 
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradientK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:џџџџџџџџџ*
T0
Ь
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Т
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
а
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1cA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ш
UA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
п
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/NegNegUA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

\A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpT^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeP^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
О
dA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
fA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ

DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:

CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
№
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addAdd;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
ќ
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modFloorModBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
в
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeRangeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/delta*

Tidx0*
_output_shapes
:

IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :

CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/FillFillFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
Љ
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill*
N*#
_output_shapes
:џџџџџџџџџ*
T0

HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/MaximumMaximumLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordivFloorDivDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
Ќ
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Љ
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
Џ
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0

DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0

CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

EA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1ProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0

IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
Ц
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/CastCastIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
С
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

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
џ
AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/add_grad/Shape3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
б
/A2S/gradients_2/A2S/best_q_network/add_grad/SumSumA2S/gradients_2/AddN_2AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
т
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
е
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
л
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
В
<A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1
О
DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/add_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
З
FA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1*
_output_shapes
:

5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ф
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Й
?A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul8^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
Ш
GA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Х
IA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1

A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Ћ
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
ж
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 

A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 

A2S/beta2_power_2/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
Ћ
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
ж
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 

A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
г
BA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
р
0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam
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
Х
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/w/AdamBA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(
и
5A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
е
DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    
т
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
Ы
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
м
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
Ы
BA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
и
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
С
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/b/AdamBA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
д
5A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:*
T0
Э
DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
к
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
Ч
9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
и
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
П
<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Ь
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
Љ
1A2S/A2S/best_q_network/LayerNorm/beta/Adam/AssignAssign*A2S/A2S/best_q_network/LayerNorm/beta/Adam<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
Т
/A2S/A2S/best_q_network/LayerNorm/beta/Adam/readIdentity*A2S/A2S/best_q_network/LayerNorm/beta/Adam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
С
>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Ю
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
Џ
3A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/AssignAssign,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zeros*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ц
1A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/readIdentity,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
С
=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
Ю
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
­
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/AssignAssign+A2S/A2S/best_q_network/LayerNorm/gamma/Adam=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Х
0A2S/A2S/best_q_network/LayerNorm/gamma/Adam/readIdentity+A2S/A2S/best_q_network/LayerNorm/gamma/Adam*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:*
T0
У
?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
а
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
Г
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
Щ
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/readIdentity-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1*
_output_shapes
:*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
г
BA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
р
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
Х
7A2S/A2S/best_q_network/best_q_network/out/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/w/AdamBA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
и
5A2S/A2S/best_q_network/best_q_network/out/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/w/Adam*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
е
DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
т
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
Ы
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
м
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
Ы
BA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
и
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
С
7A2S/A2S/best_q_network/best_q_network/out/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/b/AdamBA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
д
5A2S/A2S/best_q_network/best_q_network/out/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/b/Adam*
_output_shapes
:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
Э
DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
к
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
Ч
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
и
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
 *wО?*
dtype0
W
A2S/Adam_2/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Б
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( 
Њ
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/b0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonFA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0

=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( 
Ѕ
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
Г
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/w0A2S/A2S/best_q_network/best_q_network/out/w/Adam2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
use_nesterov( *
_output_shapes

:
Ќ
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/b0A2S/A2S/best_q_network/best_q_network/out/b/Adam2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonHA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
Џ
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
О
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
Б
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: *
T0
Т
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
е

A2S/Adam_2NoOpD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1

A2S/group_depsNoOp

A2S/group_deps_1NoOp
Ћ
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
: "ІzDП     (+дE	іbФПzжAJВЏ
тР
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
2	
ы
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
2	

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
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
	2	
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.3.02v1.3.0-rc2-20-g0787eeeї
s
A2S/observationsPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
A2S/advantagesPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
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
:џџџџџџџџџ*
shape:џџџџџџџџџ
W
A2S/average_rewardPlaceholder*
dtype0*
_output_shapes
:*
shape:
ё
VA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
у
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
у
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
р
^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes

:*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
ђ
TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes
: 

TA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w
і
PA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
ѓ
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
ы
<A2S/backup_policy_network/backup_policy_network/fc0/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/wPA2S/backup_policy_network/backup_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
№
:A2S/backup_policy_network/backup_policy_network/fc0/w/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/w*
_output_shapes

:
о
GA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ы
5A2S/backup_policy_network/backup_policy_network/fc0/b
VariableV2*
shared_name *H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
о
<A2S/backup_policy_network/backup_policy_network/fc0/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/fc0/bGA2S/backup_policy_network/backup_policy_network/fc0/b/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
ь
:A2S/backup_policy_network/backup_policy_network/fc0/b/readIdentity5A2S/backup_policy_network/backup_policy_network/fc0/b*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/fc0/b*
_output_shapes
:*
T0
а
 A2S/backup_policy_network/MatMulMatMulA2S/observations:A2S/backup_policy_network/backup_policy_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Д
A2S/backup_policy_network/addAdd A2S/backup_policy_network/MatMul:A2S/backup_policy_network/backup_policy_network/fc0/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
Ф
:A2S/backup_policy_network/LayerNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
б
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
Њ
/A2S/backup_policy_network/LayerNorm/beta/AssignAssign(A2S/backup_policy_network/LayerNorm/beta:A2S/backup_policy_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
Х
-A2S/backup_policy_network/LayerNorm/beta/readIdentity(A2S/backup_policy_network/LayerNorm/beta*
T0*;
_class1
/-loc:@A2S/backup_policy_network/LayerNorm/beta*
_output_shapes
:
Х
:A2S/backup_policy_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
valueB*  ?
г
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
­
0A2S/backup_policy_network/LayerNorm/gamma/AssignAssign)A2S/backup_policy_network/LayerNorm/gamma:A2S/backup_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Ш
.A2S/backup_policy_network/LayerNorm/gamma/readIdentity)A2S/backup_policy_network/LayerNorm/gamma*
T0*<
_class2
0.loc:@A2S/backup_policy_network/LayerNorm/gamma*
_output_shapes
:

BA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
ъ
0A2S/backup_policy_network/LayerNorm/moments/meanMeanA2S/backup_policy_network/addBA2S/backup_policy_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0
Ќ
8A2S/backup_policy_network/LayerNorm/moments/StopGradientStopGradient0A2S/backup_policy_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
н
=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_policy_network/add8A2S/backup_policy_network/LayerNorm/moments/StopGradient*'
_output_shapes
:џџџџџџџџџ*
T0

FA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

4A2S/backup_policy_network/LayerNorm/moments/varianceMean=A2S/backup_policy_network/LayerNorm/moments/SquaredDifferenceFA2S/backup_policy_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0
x
3A2S/backup_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
е
1A2S/backup_policy_network/LayerNorm/batchnorm/addAdd4A2S/backup_policy_network/LayerNorm/moments/variance3A2S/backup_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
3A2S/backup_policy_network/LayerNorm/batchnorm/RsqrtRsqrt1A2S/backup_policy_network/LayerNorm/batchnorm/add*'
_output_shapes
:џџџџџџџџџ*
T0
Я
1A2S/backup_policy_network/LayerNorm/batchnorm/mulMul3A2S/backup_policy_network/LayerNorm/batchnorm/Rsqrt.A2S/backup_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
О
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_1MulA2S/backup_policy_network/add1A2S/backup_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
б
3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2Mul0A2S/backup_policy_network/LayerNorm/moments/mean1A2S/backup_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
1A2S/backup_policy_network/LayerNorm/batchnorm/subSub-A2S/backup_policy_network/LayerNorm/beta/read3A2S/backup_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
д
3A2S/backup_policy_network/LayerNorm/batchnorm/add_1Add3A2S/backup_policy_network/LayerNorm/batchnorm/mul_11A2S/backup_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
d
A2S/backup_policy_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
Ќ
A2S/backup_policy_network/mulMulA2S/backup_policy_network/mul/x3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/backup_policy_network/AbsAbs3A2S/backup_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
f
!A2S/backup_policy_network/mul_1/xConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 

A2S/backup_policy_network/mul_1Mul!A2S/backup_policy_network/mul_1/xA2S/backup_policy_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/backup_policy_network/add_1AddA2S/backup_policy_network/mulA2S/backup_policy_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
p
+A2S/backup_policy_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ё
VA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB"      
у
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
у
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
р
^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/shape*
seed28*
dtype0*
_output_shapes

:*

seed*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w
ђ
TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/subSubTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/maxTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w

TA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulMul^A2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/RandomUniformTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
і
PA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniformAddTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/mulTA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
ѓ
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
ы
<A2S/backup_policy_network/backup_policy_network/out/w/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/wPA2S/backup_policy_network/backup_policy_network/out/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
№
:A2S/backup_policy_network/backup_policy_network/out/w/readIdentity5A2S/backup_policy_network/backup_policy_network/out/w*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/w*
_output_shapes

:
о
GA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ы
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
о
<A2S/backup_policy_network/backup_policy_network/out/b/AssignAssign5A2S/backup_policy_network/backup_policy_network/out/bGA2S/backup_policy_network/backup_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
validate_shape(*
_output_shapes
:
ь
:A2S/backup_policy_network/backup_policy_network/out/b/readIdentity5A2S/backup_policy_network/backup_policy_network/out/b*
T0*H
_class>
<:loc:@A2S/backup_policy_network/backup_policy_network/out/b*
_output_shapes
:
с
"A2S/backup_policy_network/MatMul_1MatMulA2S/backup_policy_network/add_1:A2S/backup_policy_network/backup_policy_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
И
A2S/backup_policy_network/add_2Add"A2S/backup_policy_network/MatMul_1:A2S/backup_policy_network/backup_policy_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
щ
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
л
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
л
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
д
ZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shape*
_output_shapes

:*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
seed2H*
dtype0
т
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
є
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
ц
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
ы
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
л
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(
ф
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:*
T0
ж
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0
у
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
Ю
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(
р
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
Ъ
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ќ
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
Р
8A2S/best_policy_network/LayerNorm/beta/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Э
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
Ђ
-A2S/best_policy_network/LayerNorm/beta/AssignAssign&A2S/best_policy_network/LayerNorm/beta8A2S/best_policy_network/LayerNorm/beta/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
П
+A2S/best_policy_network/LayerNorm/beta/readIdentity&A2S/best_policy_network/LayerNorm/beta*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
С
8A2S/best_policy_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*  ?
Я
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
Ѕ
.A2S/best_policy_network/LayerNorm/gamma/AssignAssign'A2S/best_policy_network/LayerNorm/gamma8A2S/best_policy_network/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma
Т
,A2S/best_policy_network/LayerNorm/gamma/readIdentity'A2S/best_policy_network/LayerNorm/gamma*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:*
T0

@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ф
.A2S/best_policy_network/LayerNorm/moments/meanMeanA2S/best_policy_network/add@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
6A2S/best_policy_network/LayerNorm/moments/StopGradientStopGradient.A2S/best_policy_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
з
;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

DA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

2A2S/best_policy_network/LayerNorm/moments/varianceMean;A2S/best_policy_network/LayerNorm/moments/SquaredDifferenceDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
v
1A2S/best_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
Я
/A2S/best_policy_network/LayerNorm/batchnorm/addAdd2A2S/best_policy_network/LayerNorm/moments/variance1A2S/best_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:џџџџџџџџџ

1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtRsqrt/A2S/best_policy_network/LayerNorm/batchnorm/add*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
/A2S/best_policy_network/LayerNorm/batchnorm/mulMul1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt,A2S/best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
И
1A2S/best_policy_network/LayerNorm/batchnorm/mul_1MulA2S/best_policy_network/add/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
1A2S/best_policy_network/LayerNorm/batchnorm/mul_2Mul.A2S/best_policy_network/LayerNorm/moments/mean/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
/A2S/best_policy_network/LayerNorm/batchnorm/subSub+A2S/best_policy_network/LayerNorm/beta/read1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
1A2S/best_policy_network/LayerNorm/batchnorm/add_1Add1A2S/best_policy_network/LayerNorm/batchnorm/mul_1/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
b
A2S/best_policy_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
І
A2S/best_policy_network/mulMulA2S/best_policy_network/mul/x1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_policy_network/AbsAbs1A2S/best_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
d
A2S/best_policy_network/mul_1/xConst*
_output_shapes
: *
valueB
 *ЭЬЬ>*
dtype0

A2S/best_policy_network/mul_1MulA2S/best_policy_network/mul_1/xA2S/best_policy_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_policy_network/add_1AddA2S/best_policy_network/mulA2S/best_policy_network/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
n
)A2S/best_policy_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
щ
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
л
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
л
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
д
ZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2u*
dtype0*
_output_shapes

:
т
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes
: 
є
PA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
ц
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:*
T0
ы
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
л
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:
ф
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:
ж
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
у
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0
Ю
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
р
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
й
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/add_16A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
А
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
э
TA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
п
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
п
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
л
\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/shape*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
seed2*
dtype0*
_output_shapes

:*

seed
ъ
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w
ќ
RA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
ю
NA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform/min*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:*
T0
я
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
у
:A2S/backup_value_network/backup_value_network/fc0/w/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/wNA2S/backup_value_network/backup_value_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
validate_shape(*
_output_shapes

:
ъ
8A2S/backup_value_network/backup_value_network/fc0/w/readIdentity3A2S/backup_value_network/backup_value_network/fc0/w*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/w*
_output_shapes

:
к
EA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zerosConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ч
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
ж
:A2S/backup_value_network/backup_value_network/fc0/b/AssignAssign3A2S/backup_value_network/backup_value_network/fc0/bEA2S/backup_value_network/backup_value_network/fc0/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
validate_shape(
ц
8A2S/backup_value_network/backup_value_network/fc0/b/readIdentity3A2S/backup_value_network/backup_value_network/fc0/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/fc0/b*
_output_shapes
:
Э
A2S/backup_value_network/MatMulMatMulA2S/observations8A2S/backup_value_network/backup_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
A2S/backup_value_network/addAddA2S/backup_value_network/MatMul8A2S/backup_value_network/backup_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
Т
9A2S/backup_value_network/LayerNorm/beta/Initializer/zerosConst*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Я
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
І
.A2S/backup_value_network/LayerNorm/beta/AssignAssign'A2S/backup_value_network/LayerNorm/beta9A2S/backup_value_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta
Т
,A2S/backup_value_network/LayerNorm/beta/readIdentity'A2S/backup_value_network/LayerNorm/beta*
T0*:
_class0
.,loc:@A2S/backup_value_network/LayerNorm/beta*
_output_shapes
:
У
9A2S/backup_value_network/LayerNorm/gamma/Initializer/onesConst*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
б
(A2S/backup_value_network/LayerNorm/gamma
VariableV2*
_output_shapes
:*
shared_name *;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
	container *
shape:*
dtype0
Љ
/A2S/backup_value_network/LayerNorm/gamma/AssignAssign(A2S/backup_value_network/LayerNorm/gamma9A2S/backup_value_network/LayerNorm/gamma/Initializer/ones*
T0*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
Х
-A2S/backup_value_network/LayerNorm/gamma/readIdentity(A2S/backup_value_network/LayerNorm/gamma*;
_class1
/-loc:@A2S/backup_value_network/LayerNorm/gamma*
_output_shapes
:*
T0

AA2S/backup_value_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
ч
/A2S/backup_value_network/LayerNorm/moments/meanMeanA2S/backup_value_network/addAA2S/backup_value_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
7A2S/backup_value_network/LayerNorm/moments/StopGradientStopGradient/A2S/backup_value_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
к
<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_value_network/add7A2S/backup_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

EA2S/backup_value_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0

3A2S/backup_value_network/LayerNorm/moments/varianceMean<A2S/backup_value_network/LayerNorm/moments/SquaredDifferenceEA2S/backup_value_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
w
2A2S/backup_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
в
0A2S/backup_value_network/LayerNorm/batchnorm/addAdd3A2S/backup_value_network/LayerNorm/moments/variance2A2S/backup_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

2A2S/backup_value_network/LayerNorm/batchnorm/RsqrtRsqrt0A2S/backup_value_network/LayerNorm/batchnorm/add*'
_output_shapes
:џџџџџџџџџ*
T0
Ь
0A2S/backup_value_network/LayerNorm/batchnorm/mulMul2A2S/backup_value_network/LayerNorm/batchnorm/Rsqrt-A2S/backup_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
Л
2A2S/backup_value_network/LayerNorm/batchnorm/mul_1MulA2S/backup_value_network/add0A2S/backup_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
2A2S/backup_value_network/LayerNorm/batchnorm/mul_2Mul/A2S/backup_value_network/LayerNorm/moments/mean0A2S/backup_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
0A2S/backup_value_network/LayerNorm/batchnorm/subSub,A2S/backup_value_network/LayerNorm/beta/read2A2S/backup_value_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
б
2A2S/backup_value_network/LayerNorm/batchnorm/add_1Add2A2S/backup_value_network/LayerNorm/batchnorm/mul_10A2S/backup_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
c
A2S/backup_value_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
Љ
A2S/backup_value_network/mulMulA2S/backup_value_network/mul/x2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/backup_value_network/AbsAbs2A2S/backup_value_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
e
 A2S/backup_value_network/mul_1/xConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 

A2S/backup_value_network/mul_1Mul A2S/backup_value_network/mul_1/xA2S/backup_value_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/backup_value_network/add_1AddA2S/backup_value_network/mulA2S/backup_value_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
o
*A2S/backup_value_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
э
TA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB"      *
dtype0
п
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *ЭЬЬН*
dtype0
п
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
л
\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformTA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/shape*
_output_shapes

:*

seed*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
seed2В*
dtype0
ъ
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/subSubRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/maxRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes
: 
ќ
RA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulMul\A2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/RandomUniformRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w
ю
NA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniformAddRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/mulRA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:
я
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
у
:A2S/backup_value_network/backup_value_network/out/w/AssignAssign3A2S/backup_value_network/backup_value_network/out/wNA2S/backup_value_network/backup_value_network/out/w/Initializer/random_uniform*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
ъ
8A2S/backup_value_network/backup_value_network/out/w/readIdentity3A2S/backup_value_network/backup_value_network/out/w*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/w*
_output_shapes

:*
T0
к
EA2S/backup_value_network/backup_value_network/out/b/Initializer/zerosConst*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ч
3A2S/backup_value_network/backup_value_network/out/b
VariableV2*
_output_shapes
:*
shared_name *F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
	container *
shape:*
dtype0
ж
:A2S/backup_value_network/backup_value_network/out/b/AssignAssign3A2S/backup_value_network/backup_value_network/out/bEA2S/backup_value_network/backup_value_network/out/b/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
validate_shape(*
_output_shapes
:
ц
8A2S/backup_value_network/backup_value_network/out/b/readIdentity3A2S/backup_value_network/backup_value_network/out/b*
T0*F
_class<
:8loc:@A2S/backup_value_network/backup_value_network/out/b*
_output_shapes
:
н
!A2S/backup_value_network/MatMul_1MatMulA2S/backup_value_network/add_18A2S/backup_value_network/backup_value_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Д
A2S/backup_value_network/add_2Add!A2S/backup_value_network/MatMul_18A2S/backup_value_network/backup_value_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
х
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
з
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
з
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Я
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2Т*
dtype0*
_output_shapes

:*

seed
к
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes
: 
ь
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
о
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
ч
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
г
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(
о
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
в
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
п
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
Ц
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
к
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:*
T0
Ч
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ј
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
О
7A2S/best_value_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0
Ы
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

,A2S/best_value_network/LayerNorm/beta/AssignAssign%A2S/best_value_network/LayerNorm/beta7A2S/best_value_network/LayerNorm/beta/Initializer/zeros*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
М
*A2S/best_value_network/LayerNorm/beta/readIdentity%A2S/best_value_network/LayerNorm/beta*
_output_shapes
:*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
П
7A2S/best_value_network/LayerNorm/gamma/Initializer/onesConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Э
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
Ё
-A2S/best_value_network/LayerNorm/gamma/AssignAssign&A2S/best_value_network/LayerNorm/gamma7A2S/best_value_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(
П
+A2S/best_value_network/LayerNorm/gamma/readIdentity&A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma

?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
с
-A2S/best_value_network/LayerNorm/moments/meanMeanA2S/best_value_network/add?A2S/best_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
І
5A2S/best_value_network/LayerNorm/moments/StopGradientStopGradient-A2S/best_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:џџџџџџџџџ
д
:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

CA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

1A2S/best_value_network/LayerNorm/moments/varianceMean:A2S/best_value_network/LayerNorm/moments/SquaredDifferenceCA2S/best_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0
u
0A2S/best_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
Ь
.A2S/best_value_network/LayerNorm/batchnorm/addAdd1A2S/best_value_network/LayerNorm/moments/variance0A2S/best_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

0A2S/best_value_network/LayerNorm/batchnorm/RsqrtRsqrt.A2S/best_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
.A2S/best_value_network/LayerNorm/batchnorm/mulMul0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt+A2S/best_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
Е
0A2S/best_value_network/LayerNorm/batchnorm/mul_1MulA2S/best_value_network/add.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
0A2S/best_value_network/LayerNorm/batchnorm/mul_2Mul-A2S/best_value_network/LayerNorm/moments/mean.A2S/best_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Х
.A2S/best_value_network/LayerNorm/batchnorm/subSub*A2S/best_value_network/LayerNorm/beta/read0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
0A2S/best_value_network/LayerNorm/batchnorm/add_1Add0A2S/best_value_network/LayerNorm/batchnorm/mul_1.A2S/best_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
a
A2S/best_value_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 
Ѓ
A2S/best_value_network/mulMulA2S/best_value_network/mul/x0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_value_network/AbsAbs0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
c
A2S/best_value_network/mul_1/xConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 

A2S/best_value_network/mul_1MulA2S/best_value_network/mul_1/xA2S/best_value_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/best_value_network/add_1AddA2S/best_value_network/mulA2S/best_value_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
m
(A2S/best_value_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
х
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"      *
dtype0
з
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *ЭЬЬН
з
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Я
XA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2я
к
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes
: 
ь
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
о
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
ч
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
г
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
о
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
в
AA2S/best_value_network/best_value_network/out/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
п
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
Ц
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
к
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
е
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/add_14A2S/best_value_network/best_value_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ќ
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
b
A2S/Reshape/shapeConst*
_output_shapes
:*
valueB"џџџџ   *
dtype0

A2S/ReshapeReshapeA2S/backup_policy_network/add_2A2S/Reshape/shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
d
A2S/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

A2S/Reshape_1ReshapeA2S/best_policy_network/add_2A2S/Reshape_1/shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
N
	A2S/ConstConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
P
A2S/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   ?
Y
A2S/Normal/locIdentityA2S/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
H
A2S/Normal/scaleIdentity	A2S/Const*
T0*
_output_shapes
: 
]
A2S/Normal_1/locIdentityA2S/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
L
A2S/Normal_1/scaleIdentityA2S/Const_1*
T0*
_output_shapes
: 
o
*A2S/KullbackLeibler/kl_normal_normal/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
Д
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 

(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal_1/locA2S/Normal/loc*
T0*'
_output_shapes
:џџџџџџџџџ

-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:џџџџџџџџџ
­
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0
Ф
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*
_output_shapes
: *
T0
~
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
_output_shapes
: *
T0
Ј
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
T0*
_output_shapes
: 
Ќ
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
T0*
_output_shapes
: 
Н
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
\
A2S/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:

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
Г
-A2S/Normal_2/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_2/batch_shape_tensor/Shape'A2S/Normal_2/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
]
A2S/concat/values_0Const*
_output_shapes
:*
valueB:*
dtype0
Q
A2S/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ

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
 *  ?*
dtype0*
_output_shapes
: 
А
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
seed2Ё*

seed

A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
t
A2S/mulMulA2S/random_normalA2S/Normal_1/scale*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
h
A2S/addAddA2S/mulA2S/Normal_1/loc*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
h
A2S/Reshape_2/shapeConst*!
valueB"џџџџ      *
dtype0*
_output_shapes
:
z
A2S/Reshape_2ReshapeA2S/addA2S/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:џџџџџџџџџ
S
A2S/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 

A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*
N*'
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
н
LA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
Я
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/minConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
Я
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
У
TA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
seed2­
Ъ
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes
: *
T0
м
JA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:
Ю
FA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
п
+A2S/backup_q_network/backup_q_network/fc0/w
VariableV2*
_output_shapes

:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
	container *
shape
:*
dtype0
У
2A2S/backup_q_network/backup_q_network/fc0/w/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/wFA2S/backup_q_network/backup_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w*
validate_shape(*
_output_shapes

:
в
0A2S/backup_q_network/backup_q_network/fc0/w/readIdentity+A2S/backup_q_network/backup_q_network/fc0/w*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/w
Ъ
=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
з
+A2S/backup_q_network/backup_q_network/fc0/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b*
	container 
Ж
2A2S/backup_q_network/backup_q_network/fc0/b/AssignAssign+A2S/backup_q_network/backup_q_network/fc0/b=A2S/backup_q_network/backup_q_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b
Ю
0A2S/backup_q_network/backup_q_network/fc0/b/readIdentity+A2S/backup_q_network/backup_q_network/fc0/b*
_output_shapes
:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/fc0/b
Н
A2S/backup_q_network/MatMulMatMulA2S/concat_10A2S/backup_q_network/backup_q_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
 
A2S/backup_q_network/addAddA2S/backup_q_network/MatMul0A2S/backup_q_network/backup_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
К
5A2S/backup_q_network/LayerNorm/beta/Initializer/zerosConst*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Ч
#A2S/backup_q_network/LayerNorm/beta
VariableV2*
shared_name *6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes
:

*A2S/backup_q_network/LayerNorm/beta/AssignAssign#A2S/backup_q_network/LayerNorm/beta5A2S/backup_q_network/LayerNorm/beta/Initializer/zeros*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(
Ж
(A2S/backup_q_network/LayerNorm/beta/readIdentity#A2S/backup_q_network/LayerNorm/beta*
_output_shapes
:*
T0*6
_class,
*(loc:@A2S/backup_q_network/LayerNorm/beta
Л
5A2S/backup_q_network/LayerNorm/gamma/Initializer/onesConst*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Щ
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

+A2S/backup_q_network/LayerNorm/gamma/AssignAssign$A2S/backup_q_network/LayerNorm/gamma5A2S/backup_q_network/LayerNorm/gamma/Initializer/ones*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Й
)A2S/backup_q_network/LayerNorm/gamma/readIdentity$A2S/backup_q_network/LayerNorm/gamma*
_output_shapes
:*
T0*7
_class-
+)loc:@A2S/backup_q_network/LayerNorm/gamma

=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
л
+A2S/backup_q_network/LayerNorm/moments/meanMeanA2S/backup_q_network/add=A2S/backup_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
Ђ
3A2S/backup_q_network/LayerNorm/moments/StopGradientStopGradient+A2S/backup_q_network/LayerNorm/moments/mean*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/backup_q_network/add3A2S/backup_q_network/LayerNorm/moments/StopGradient*'
_output_shapes
:џџџџџџџџџ*
T0

AA2S/backup_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

/A2S/backup_q_network/LayerNorm/moments/varianceMean8A2S/backup_q_network/LayerNorm/moments/SquaredDifferenceAA2S/backup_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
s
.A2S/backup_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
Ц
,A2S/backup_q_network/LayerNorm/batchnorm/addAdd/A2S/backup_q_network/LayerNorm/moments/variance.A2S/backup_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:џџџџџџџџџ*
T0

.A2S/backup_q_network/LayerNorm/batchnorm/RsqrtRsqrt,A2S/backup_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
Р
,A2S/backup_q_network/LayerNorm/batchnorm/mulMul.A2S/backup_q_network/LayerNorm/batchnorm/Rsqrt)A2S/backup_q_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
Џ
.A2S/backup_q_network/LayerNorm/batchnorm/mul_1MulA2S/backup_q_network/add,A2S/backup_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Т
.A2S/backup_q_network/LayerNorm/batchnorm/mul_2Mul+A2S/backup_q_network/LayerNorm/moments/mean,A2S/backup_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
П
,A2S/backup_q_network/LayerNorm/batchnorm/subSub(A2S/backup_q_network/LayerNorm/beta/read.A2S/backup_q_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
Х
.A2S/backup_q_network/LayerNorm/batchnorm/add_1Add.A2S/backup_q_network/LayerNorm/batchnorm/mul_1,A2S/backup_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
_
A2S/backup_q_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 

A2S/backup_q_network/mulMulA2S/backup_q_network/mul/x.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/backup_q_network/AbsAbs.A2S/backup_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
a
A2S/backup_q_network/mul_1/xConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 

A2S/backup_q_network/mul_1MulA2S/backup_q_network/mul_1/xA2S/backup_q_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/backup_q_network/add_1AddA2S/backup_q_network/mulA2S/backup_q_network/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
k
&A2S/backup_q_network/dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
н
LA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
Я
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *ЭЬЬН
Я
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
У
TA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformLA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/shape*
_output_shapes

:*

seed*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
seed2к*
dtype0
Ъ
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/subSubJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/maxJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes
: 
м
JA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulMulTA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/RandomUniformJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
Ю
FA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniformAddJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/mulJA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:
п
+A2S/backup_q_network/backup_q_network/out/w
VariableV2*
shared_name *>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
	container *
shape
:*
dtype0*
_output_shapes

:
У
2A2S/backup_q_network/backup_q_network/out/w/AssignAssign+A2S/backup_q_network/backup_q_network/out/wFA2S/backup_q_network/backup_q_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w
в
0A2S/backup_q_network/backup_q_network/out/w/readIdentity+A2S/backup_q_network/backup_q_network/out/w*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/w*
_output_shapes

:*
T0
Ъ
=A2S/backup_q_network/backup_q_network/out/b/Initializer/zerosConst*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
з
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
Ж
2A2S/backup_q_network/backup_q_network/out/b/AssignAssign+A2S/backup_q_network/backup_q_network/out/b=A2S/backup_q_network/backup_q_network/out/b/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
validate_shape(*
_output_shapes
:
Ю
0A2S/backup_q_network/backup_q_network/out/b/readIdentity+A2S/backup_q_network/backup_q_network/out/b*
T0*>
_class4
20loc:@A2S/backup_q_network/backup_q_network/out/b*
_output_shapes
:
Э
A2S/backup_q_network/MatMul_1MatMulA2S/backup_q_network/add_10A2S/backup_q_network/backup_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Є
A2S/backup_q_network/add_2AddA2S/backup_q_network/MatMul_10A2S/backup_q_network/backup_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
е
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"      *
dtype0*
_output_shapes
:
Ч
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
Ч
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
З
PA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2ъ
К
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes
: 
Ь
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
О
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
з
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
Г
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
Ц
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
Т
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    
Я
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
І
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
Т
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
З
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
3A2S/best_q_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0
У
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

(A2S/best_q_network/LayerNorm/beta/AssignAssign!A2S/best_q_network/LayerNorm/beta3A2S/best_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
А
&A2S/best_q_network/LayerNorm/beta/readIdentity!A2S/best_q_network/LayerNorm/beta*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
З
3A2S/best_q_network/LayerNorm/gamma/Initializer/onesConst*
_output_shapes
:*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*  ?*
dtype0
Х
"A2S/best_q_network/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma

)A2S/best_q_network/LayerNorm/gamma/AssignAssign"A2S/best_q_network/LayerNorm/gamma3A2S/best_q_network/LayerNorm/gamma/Initializer/ones*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Г
'A2S/best_q_network/LayerNorm/gamma/readIdentity"A2S/best_q_network/LayerNorm/gamma*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:

;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
е
)A2S/best_q_network/LayerNorm/moments/meanMeanA2S/best_q_network/add;A2S/best_q_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ

1A2S/best_q_network/LayerNorm/moments/StopGradientStopGradient)A2S/best_q_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
6A2S/best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferenceA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradient*'
_output_shapes
:џџџџџџџџџ*
T0

?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
§
-A2S/best_q_network/LayerNorm/moments/varianceMean6A2S/best_q_network/LayerNorm/moments/SquaredDifference?A2S/best_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
q
,A2S/best_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *ЬМ+*
dtype0*
_output_shapes
: 
Р
*A2S/best_q_network/LayerNorm/batchnorm/addAdd-A2S/best_q_network/LayerNorm/moments/variance,A2S/best_q_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:џџџџџџџџџ

,A2S/best_q_network/LayerNorm/batchnorm/RsqrtRsqrt*A2S/best_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:џџџџџџџџџ
К
*A2S/best_q_network/LayerNorm/batchnorm/mulMul,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
,A2S/best_q_network/LayerNorm/batchnorm/mul_1MulA2S/best_q_network/add*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
М
,A2S/best_q_network/LayerNorm/batchnorm/mul_2Mul)A2S/best_q_network/LayerNorm/moments/mean*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Й
*A2S/best_q_network/LayerNorm/batchnorm/subSub&A2S/best_q_network/LayerNorm/beta/read,A2S/best_q_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:џџџџџџџџџ*
T0
П
,A2S/best_q_network/LayerNorm/batchnorm/add_1Add,A2S/best_q_network/LayerNorm/batchnorm/mul_1*A2S/best_q_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
]
A2S/best_q_network/mul/xConst*
valueB
 *?*
dtype0*
_output_shapes
: 

A2S/best_q_network/mulMulA2S/best_q_network/mul/x,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
}
A2S/best_q_network/AbsAbs,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
_
A2S/best_q_network/mul_1/xConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 

A2S/best_q_network/mul_1MulA2S/best_q_network/mul_1/xA2S/best_q_network/Abs*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/best_q_network/add_1AddA2S/best_q_network/mulA2S/best_q_network/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
i
$A2S/best_q_network/dropout/keep_probConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
е
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"      *
dtype0*
_output_shapes
:
Ч
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
Ч
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0
З
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
seed2*
dtype0*
_output_shapes

:
К
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes
: 
Ь
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
О
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
з
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
Г
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
Ц
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:*
T0
Т
9A2S/best_q_network/best_q_network/out/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
Я
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
І
.A2S/best_q_network/best_q_network/out/b/AssignAssign'A2S/best_q_network/best_q_network/out/b9A2S/best_q_network/best_q_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
Т
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
Х
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/add_1,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
}
%A2S/Normal_3/log_prob/standardize/subSubA2S/actionsA2S/Normal_1/loc*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
)A2S/Normal_3/log_prob/standardize/truedivRealDiv%A2S/Normal_3/log_prob/standardize/subA2S/Normal_1/scale*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/Normal_3/log_prob/SquareSquare)A2S/Normal_3/log_prob/standardize/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
`
A2S/Normal_3/log_prob/mul/xConst*
valueB
 *   П*
dtype0*
_output_shapes
: 

A2S/Normal_3/log_prob/mulMulA2S/Normal_3/log_prob/mul/xA2S/Normal_3/log_prob/Square*'
_output_shapes
:џџџџџџџџџ*
T0
U
A2S/Normal_3/log_prob/LogLogA2S/Normal_1/scale*
T0*
_output_shapes
: 
`
A2S/Normal_3/log_prob/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 
y
A2S/Normal_3/log_prob/addAddA2S/Normal_3/log_prob/add/xA2S/Normal_3/log_prob/Log*
_output_shapes
: *
T0

A2S/Normal_3/log_prob/subSubA2S/Normal_3/log_prob/mulA2S/Normal_3/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
[
A2S/NegNegA2S/Normal_3/log_prob/sub*
T0*'
_output_shapes
:џџџџџџџџџ
[
	A2S/mul_1MulA2S/NegA2S/advantages*
T0*'
_output_shapes
:џџџџџџџџџ
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
dtype0*
_output_shapes
:*
valueB"       
h

A2S/Mean_2Mean	A2S/mul_1A2S/Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
t
A2S/policy_network_loss/tagsConst*
_output_shapes
: *(
valueB BA2S/policy_network_loss*
dtype0
s
A2S/policy_network_lossScalarSummaryA2S/policy_network_loss/tags
A2S/Mean_2*
_output_shapes
: *
T0

A2S/SquaredDifferenceSquaredDifferenceA2S/best_value_network/add_2A2S/returns*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_5Const*
dtype0*
_output_shapes
:*
valueB"       
t

A2S/Mean_3MeanA2S/SquaredDifferenceA2S/Const_5*
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
A2S/Mean_3*
_output_shapes
: *
T0

A2S/SquaredDifference_1SquaredDifferenceA2S/best_q_network/add_2A2S/returns*
T0*'
_output_shapes
:џџџџџџџџџ
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
A2S/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
A2S/gradients/ConstConst*
valueB
 *  ?*
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
Ј
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
К
"A2S/gradients/A2S/Mean_2_grad/TileTile%A2S/gradients/A2S/Mean_2_grad/Reshape#A2S/gradients/A2S/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
Д
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
И
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
 
%A2S/gradients/A2S/Mean_2_grad/MaximumMaximum$A2S/gradients/A2S/Mean_2_grad/Prod_1'A2S/gradients/A2S/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

&A2S/gradients/A2S/Mean_2_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_2_grad/Prod%A2S/gradients/A2S/Mean_2_grad/Maximum*
_output_shapes
: *
T0

"A2S/gradients/A2S/Mean_2_grad/CastCast&A2S/gradients/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Њ
%A2S/gradients/A2S/Mean_2_grad/truedivRealDiv"A2S/gradients/A2S/Mean_2_grad/Tile"A2S/gradients/A2S/Mean_2_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
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
в
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_2_grad/truedivA2S/advantages*'
_output_shapes
:џџџџџџџџџ*
T0
Н
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Е
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
У
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1

5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape

7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ShapeShapeA2S/Normal_3/log_prob/mul*
out_type0*
_output_shapes
:*
T0
w
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

BA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
л
0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
п
2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_3/log_prob/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Sum_1*
_output_shapes
:*
T0
и
6A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Е
=A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape_1
Т
EA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/sub_grad/Reshape
З
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

4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1ShapeA2S/Normal_3/log_prob/Square*
T0*
out_type0*
_output_shapes
:

BA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ю
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_3/log_prob/Square*
T0*'
_output_shapes
:џџџџџџџџџ
э
0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
д
4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Я
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1MulA2S/Normal_3/log_prob/mul/xEA2S/gradients/A2S/Normal_3/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_3/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Е
=A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1
Б
EA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape*
_output_shapes
: 
Ш
GA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_3/log_prob/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ф
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *   @*
dtype0
Ю
3A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul/x)A2S/Normal_3/log_prob/standardize/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
ь
5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_3/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_3/log_prob/standardize/sub*
T0*
out_type0*
_output_shapes
:

DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
В
RA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1A2S/Normal_1/scale*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
 
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_3/log_prob/standardize/sub*'
_output_shapes
:џџџџџџџџџ*
T0
й
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/NegA2S/Normal_1/scale*
T0*'
_output_shapes
:џџџџџџџџџ
п
FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal_1/scale*
T0*'
_output_shapes
:џџџџџџџџџ
ј
@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_3/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
BA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

FA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
х
MA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1

UA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ї
WA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 

>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
T0*
out_type0*
_output_shapes
:

@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal_1/loc*
T0*
out_type0*
_output_shapes
:
І
NA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Њ
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ў
>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_3/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
І
<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:

BA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
й
IA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1
ђ
QA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ј
SA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/group_deps*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

&A2S/gradients/A2S/Reshape_1_grad/ShapeShapeA2S/best_policy_network/add_2*
_output_shapes
:*
T0*
out_type0
№
(A2S/gradients/A2S/Reshape_1_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_3/log_prob/standardize/sub_grad/tuple/control_dependency_1&A2S/gradients/A2S/Reshape_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

6A2S/gradients/A2S/best_policy_network/add_2_grad/ShapeShape A2S/best_policy_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

FA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape8A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
э
4A2S/gradients/A2S/best_policy_network/add_2_grad/SumSum(A2S/gradients/A2S/Reshape_1_grad/ReshapeFA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ё
8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_2_grad/Sum6A2S/gradients/A2S/best_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_1Sum(A2S/gradients/A2S/Reshape_1_grad/ReshapeHA2S/gradients/A2S/best_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_2_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
С
AA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1
в
IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_2_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape
Ы
KA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_2_grad/Reshape_1

:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulMatMulIA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
џ
<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/best_policy_network/add_1IA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Ш
DA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul=^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1
м
LA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMulE^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul
й
NA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1E^A2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients/A2S/best_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:

6A2S/gradients/A2S/best_policy_network/add_1_grad/ShapeShapeA2S/best_policy_network/mul*
_output_shapes
:*
T0*
out_type0

8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1ShapeA2S/best_policy_network/mul_1*
T0*
out_type0*
_output_shapes
:

FA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape8A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

4A2S/gradients/A2S/best_policy_network/add_1_grad/SumSumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyFA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ё
8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/add_1_grad/Sum6A2S/gradients/A2S/best_policy_network/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_1SumLA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependencyHA2S/gradients/A2S/best_policy_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/add_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
С
AA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1
в
IA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/add_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
и
KA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/add_1_grad/tuple/group_deps*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
w
4A2S/gradients/A2S/best_policy_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ї
6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

DA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/mul_grad/Shape6A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
щ
2A2S/gradients/A2S/best_policy_network/mul_grad/mulMulIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
2A2S/gradients/A2S/best_policy_network/mul_grad/SumSum2A2S/gradients/A2S/best_policy_network/mul_grad/mulDA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
к
6A2S/gradients/A2S/best_policy_network/mul_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/mul_grad/Sum4A2S/gradients/A2S/best_policy_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
з
4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1MulA2S/best_policy_network/mul/xIA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
љ
4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_1Sum4A2S/gradients/A2S/best_policy_network/mul_grad/mul_1FA2S/gradients/A2S/best_policy_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ё
8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/mul_grad/Sum_16A2S/gradients/A2S/best_policy_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Л
?A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape9^A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1
Й
GA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/mul_grad/Reshape@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape*
_output_shapes
: 
а
IA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
y
6A2S/gradients/A2S/best_policy_network/mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 

8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1ShapeA2S/best_policy_network/Abs*
_output_shapes
:*
T0*
out_type0

FA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape8A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
з
4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulMulKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1A2S/best_policy_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ
љ
4A2S/gradients/A2S/best_policy_network/mul_1_grad/SumSum4A2S/gradients/A2S/best_policy_network/mul_1_grad/mulFA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
р
8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeReshape4A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
н
6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1MulA2S/best_policy_network/mul_1/xKA2S/gradients/A2S/best_policy_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
џ
6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_1Sum6A2S/gradients/A2S/best_policy_network/mul_1_grad/mul_1HA2S/gradients/A2S/best_policy_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1Reshape6A2S/gradients/A2S/best_policy_network/mul_1_grad/Sum_18A2S/gradients/A2S/best_policy_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
С
AA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape;^A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1
С
IA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/mul_1_grad/ReshapeB^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape*
_output_shapes
: 
и
KA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1B^A2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
3A2S/gradients/A2S/best_policy_network/Abs_grad/SignSign1A2S/best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
э
2A2S/gradients/A2S/best_policy_network/Abs_grad/mulMulKA2S/gradients/A2S/best_policy_network/mul_1_grad/tuple/control_dependency_13A2S/gradients/A2S/best_policy_network/Abs_grad/Sign*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
A2S/gradients/AddNAddNIA2S/gradients/A2S/best_policy_network/mul_grad/tuple/control_dependency_12A2S/gradients/A2S/best_policy_network/Abs_grad/mul*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/mul_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Л
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
Ъ
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients/AddNZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
­
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients/AddN\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
§
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
Ђ
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ѕ
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
Л
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Ъ
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Е
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
­
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
џ
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_policy_network/add]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Г
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
§
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
Ђ
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape1A2S/best_policy_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
Ф
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ш
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumSum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes
:*
T0*
Tshape0
Ь
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
К
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegNegHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
Ћ
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/NegJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ї
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1

[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
 
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
И
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape.A2S/best_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
Л
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape/A2S/best_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Ъ
ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulMul]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1/A2S/best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Е
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mulZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
­
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul.A2S/best_policy_network/LayerNorm/moments/mean]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
Л
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1\A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
§
UA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeO^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
Ђ
]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeV^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1V^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
ќ
A2S/gradients/AddN_1AddN_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1_A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Й
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients/AddN_1,A2S/best_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumSumFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mulXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ї
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul1A2S/best_policy_network/LayerNorm/batchnorm/RsqrtA2S/gradients/AddN_1*'
_output_shapes
:џџџџџџџџџ*
T0
Е
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/mul_1ZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
 
LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
ї
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1

[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1

NA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad1A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
К
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape2A2S/best_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
XA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
З
FA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumSumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradXA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ї
JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeFA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/SumHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Л
HA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumNA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradZA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

LA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeHA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Sum_1JA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ї
SA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpK^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeM^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1

[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityJA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/ReshapeT^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityLA2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1T^A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
Ц
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeShape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0

JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeConst*
_output_shapes
: *
value	B :*
dtype0

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addAddDA2S/best_policy_network/LayerNorm/moments/variance/reduction_indicesJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modFloorModIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/addJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ю
KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeRangeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/startJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/SizeQA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0

PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0

JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/FillFillMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_1PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill/value*
T0*
_output_shapes
:
Ь
SA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/rangeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/modKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ќ
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumSA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitchOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0

NA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ShapeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
Л
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeReshape[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencySA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
О
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileTileMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ReshapeNA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ш
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape;A2S/best_policy_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
П
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape2A2S/best_policy_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0

KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ќ
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_2KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
А
LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Shape_3MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1MaximumLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Prod_1QA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0

PA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/ProdOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*
_output_shapes
: 
д
JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/CastCastPA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Ђ
MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truedivRealDivJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/TileJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
Ь
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape6A2S/best_policy_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
ш
dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ъ
UA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Б
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulUA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
А
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_policy_network/add6A2S/best_policy_network/LayerNorm/moments/StopGradientN^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Е
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mulRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
е
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1dA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ы
VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
й
TA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1fA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
XA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeTA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1VA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
х
RA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegXA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

_A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpW^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeS^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ъ
gA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityVA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*i
_class_
][loc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ф
iA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityRA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg`^A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*e
_class[
YWloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
Ђ
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_policy_network/add*
_output_shapes
:*
T0*
out_type0

FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addAdd@A2S/best_policy_network/LayerNorm/moments/mean/reduction_indicesFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

EA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modFloorModEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/addFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
о
GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeRangeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/startFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/SizeMA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0

LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/FillFillIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_1LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
И
OA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/rangeEA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/modGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Fill*
N*#
_output_shapes
:џџџџџџџџџ*
T0

KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/MaximumMaximumOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitchKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0

JA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivGA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ShapeIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
Е
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeReshape]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyOA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
В
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileTileIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ReshapeJA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0
Є
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_policy_network/add*
T0*
out_type0*
_output_shapes
:
З
IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape.A2S/best_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0

GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
 
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_2GA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Є
HA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Shape_3IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :

KA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumHA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Prod_1MA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 

LA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/ProdKA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*
_output_shapes
: 
Ь
FA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/CastCastLA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0

IA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truedivRealDivFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/TileFA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
A2S/gradients/AddN_2AddN]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencygA2S/gradients/A2S/best_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients/A2S/best_policy_network/LayerNorm/moments/mean_grad/truediv*
T0*_
_classU
SQloc:@A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

4A2S/gradients/A2S/best_policy_network/add_grad/ShapeShapeA2S/best_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:

6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients/A2S/best_policy_network/add_grad/Shape6A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
е
2A2S/gradients/A2S/best_policy_network/add_grad/SumSumA2S/gradients/AddN_2DA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
6A2S/gradients/A2S/best_policy_network/add_grad/ReshapeReshape2A2S/gradients/A2S/best_policy_network/add_grad/Sum4A2S/gradients/A2S/best_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
й
4A2S/gradients/A2S/best_policy_network/add_grad/Sum_1SumA2S/gradients/AddN_2FA2S/gradients/A2S/best_policy_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ф
8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1Reshape4A2S/gradients/A2S/best_policy_network/add_grad/Sum_16A2S/gradients/A2S/best_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Л
?A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_depsNoOp7^A2S/gradients/A2S/best_policy_network/add_grad/Reshape9^A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1
Ъ
GA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients/A2S/best_policy_network/add_grad/Reshape@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
У
IA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1@^A2S/gradients/A2S/best_policy_network/add_grad/tuple/group_deps*
_output_shapes
:*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/add_grad/Reshape_1

8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulMatMulGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ю
:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsGA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Т
BA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul;^A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1
д
JA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMulC^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul
б
LA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1C^A2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients/A2S/best_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:

A2S/beta1_power/initial_valueConst*
valueB
 *fff?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Ў
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
е
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(

A2S/beta1_power/readIdentityA2S/beta1_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 

A2S/beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wО?*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
dtype0
Ў
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
е
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

A2S/beta2_power/readIdentityA2S/beta2_power*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
ч
LA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
є
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
э
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/w/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
і
?A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
щ
NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
і
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
ѓ
CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:
њ
AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:
п
LA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    
ь
:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container 
щ
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/fc0/b/AdamLA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:
ђ
?A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
с
NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB*    *
dtype0
ю
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
я
CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
і
AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:
Щ
AA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zerosConst*
_output_shapes
:*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0
ж
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
Н
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/AssignAssign/A2S/A2S/best_policy_network/LayerNorm/beta/AdamAA2S/A2S/best_policy_network/LayerNorm/beta/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(
б
4A2S/A2S/best_policy_network/LayerNorm/beta/Adam/readIdentity/A2S/A2S/best_policy_network/LayerNorm/beta/Adam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
Ы
CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
и
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
У
8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/AssignAssign1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1CA2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
е
6A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/readIdentity1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
:
Ы
BA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
и
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
С
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/AssignAssign0A2S/A2S/best_policy_network/LayerNorm/gamma/AdamBA2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
д
5A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/readIdentity0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
Э
DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
к
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
Ч
9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/AssignAssign2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1DA2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
и
7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/readIdentity2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1*
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
_output_shapes
:
ч
LA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
є
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
э
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/w/AdamLA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:*
use_locking(
і
?A2S/A2S/best_policy_network/best_policy_network/out/w/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
щ
NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
і
<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1
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
ѓ
CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(
њ
AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/readIdentity<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1*
_output_shapes

:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
п
LA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ь
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
щ
AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAssign:A2S/A2S/best_policy_network/best_policy_network/out/b/AdamLA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
ђ
?A2S/A2S/best_policy_network/best_policy_network/out/b/Adam/readIdentity:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
с
NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ю
<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
я
CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignAssign<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1NA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
і
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
 *wО?*
dtype0*
_output_shapes
: 
U
A2S/Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0
к
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/w:A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/best_policy_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:
г
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/fc0/b:A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonIA2S/gradients/A2S/best_policy_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:
Ў
@A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam&A2S/best_policy_network/LayerNorm/beta/A2S/A2S/best_policy_network/LayerNorm/beta/Adam1A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon[A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
Е
AA2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam'A2S/best_policy_network/LayerNorm/gamma0A2S/A2S/best_policy_network/LayerNorm/gamma/Adam2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilon]A2S/gradients/A2S/best_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
м
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/w:A2S/A2S/best_policy_network/best_policy_network/out/w/Adam<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/best_policy_network/MatMul_1_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
use_nesterov( *
_output_shapes

:*
use_locking( 
е
KA2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam	ApplyAdam1A2S/best_policy_network/best_policy_network/out/b:A2S/A2S/best_policy_network/best_policy_network/out/b/Adam<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonKA2S/gradients/A2S/best_policy_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
д
A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta
Н
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
ж
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2L^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/fc0/b/ApplyAdamA^A2S/Adam/update_A2S/best_policy_network/LayerNorm/beta/ApplyAdamB^A2S/Adam/update_A2S/best_policy_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/w/ApplyAdamL^A2S/Adam/update_A2S/best_policy_network/best_policy_network/out/b/ApplyAdam*
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
_output_shapes
: 
С
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
ѕ
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
 *  ?*
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
Ў
'A2S/gradients_1/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_1/Fill-A2S/gradients_1/A2S/Mean_3_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
%A2S/gradients_1/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference*
_output_shapes
:*
T0*
out_type0
Р
$A2S/gradients_1/A2S/Mean_3_grad/TileTile'A2S/gradients_1/A2S/Mean_3_grad/Reshape%A2S/gradients_1/A2S/Mean_3_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
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
К
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
О
&A2S/gradients_1/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_3_grad/Shape_2'A2S/gradients_1/A2S/Mean_3_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_1/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'A2S/gradients_1/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_3_grad/Prod_1)A2S/gradients_1/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
Є
(A2S/gradients_1/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_3_grad/Prod'A2S/gradients_1/A2S/Mean_3_grad/Maximum*
T0*
_output_shapes
: 

$A2S/gradients_1/A2S/Mean_3_grad/CastCast(A2S/gradients_1/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_1/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_3_grad/Tile$A2S/gradients_1/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

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
ќ
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
 
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
У
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
М
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/best_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
щ
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
п
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
э
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

.A2S/gradients_1/A2S/SquaredDifference_grad/NegNeg4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
;A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_depsNoOp3^A2S/gradients_1/A2S/SquaredDifference_grad/Reshape/^A2S/gradients_1/A2S/SquaredDifference_grad/Neg
К
CA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIdentity2A2S/gradients_1/A2S/SquaredDifference_grad/Reshape<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*E
_class;
97loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Reshape
Д
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients_1/A2S/best_value_network/add_2_grad/ShapeShapeA2S/best_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

GA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5A2S/gradients_1/A2S/best_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
э
;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_2_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
Ф
BA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1
ж
JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_2_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Я
LA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/group_deps*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_2_grad/Reshape_1*
_output_shapes
:*
T0

;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulMatMulJA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1MatMulA2S/best_value_network/add_1JA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
Ы
EA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_depsNoOp<^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul>^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1
р
MA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIdentity;A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMulF^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
н
OA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1Identity=A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1F^A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/group_deps*P
_classF
DBloc:@A2S/gradients_1/A2S/best_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:*
T0

7A2S/gradients_1/A2S/best_value_network/add_1_grad/ShapeShapeA2S/best_value_network/mul*
out_type0*
_output_shapes
:*
T0

9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1ShapeA2S/best_value_network/mul_1*
_output_shapes
:*
T0*
out_type0

GA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5A2S/gradients_1/A2S/best_value_network/add_1_grad/SumSumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyGA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_1SumMA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependencyIA2S/gradients_1/A2S/best_value_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
њ
;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/add_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
BA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1
ж
JA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/add_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
м
LA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
x
5A2S/gradients_1/A2S/best_value_network/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Ї
7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

EA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape7A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
3A2S/gradients_1/A2S/best_value_network/mul_grad/mulMulJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
і
3A2S/gradients_1/A2S/best_value_network/mul_grad/SumSum3A2S/gradients_1/A2S/best_value_network/mul_grad/mulEA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
н
7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/mul_grad/Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
и
5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1MulA2S/best_value_network/mul/xJA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
ќ
5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_1Sum5A2S/gradients_1/A2S/best_value_network/mul_grad/mul_1GA2S/gradients_1/A2S/best_value_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/mul_grad/Sum_17A2S/gradients_1/A2S/best_value_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
О
@A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1
Н
HA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/mul_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape
д
JA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
z
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1ShapeA2S/best_value_network/Abs*
_output_shapes
:*
T0*
out_type0

GA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape9A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
и
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulMulLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1A2S/best_value_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ
ќ
5A2S/gradients_1/A2S/best_value_network/mul_1_grad/SumSum5A2S/gradients_1/A2S/best_value_network/mul_1_grad/mulGA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
у
9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeReshape5A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
о
7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1MulA2S/best_value_network/mul_1/xLA2S/gradients_1/A2S/best_value_network/add_1_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0

7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_1Sum7A2S/gradients_1/A2S/best_value_network/mul_1_grad/mul_1IA2S/gradients_1/A2S/best_value_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
њ
;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1Reshape7A2S/gradients_1/A2S/best_value_network/mul_1_grad/Sum_19A2S/gradients_1/A2S/best_value_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
BA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape<^A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1
Х
JA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/mul_1_grad/ReshapeC^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape
м
LA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1C^A2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
4A2S/gradients_1/A2S/best_value_network/Abs_grad/SignSign0A2S/best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
№
3A2S/gradients_1/A2S/best_value_network/Abs_grad/mulMulLA2S/gradients_1/A2S/best_value_network/mul_1_grad/tuple/control_dependency_14A2S/gradients_1/A2S/best_value_network/Abs_grad/Sign*'
_output_shapes
:џџџџџџџџџ*
T0
І
A2S/gradients_1/AddNAddNJA2S/gradients_1/A2S/best_value_network/mul_grad/tuple/control_dependency_13A2S/gradients_1/A2S/best_value_network/Abs_grad/mul*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/mul_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Л
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
Л
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/sub*
_output_shapes
:*
T0*
out_type0
Э
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_1/AddN[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
А
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_1/AddN]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
І
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ѕ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
Л
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
Э
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
И
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_value_network/add^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
О
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
І
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ќ
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Л
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0A2S/best_value_network/LayerNorm/batchnorm/mul_2*
out_type0*
_output_shapes
:*
T0
Ч
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ы
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumSum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes
:*
T0*
Tshape0
Я
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
М
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegNegIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
Ў
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/NegKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
њ
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1

\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:
Є
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1
И
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-A2S/best_value_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
Л
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
Э
[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mulMul^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.A2S/best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
И
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-A2S/best_value_network/LayerNorm/moments/mean^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
О
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1]A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

VA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeP^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
І
^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeW^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ќ
`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1W^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

A2S/gradients_1/AddN_1AddN`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1`A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Й
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeShape0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ч
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Х
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_1/AddN_1+A2S/best_value_network/LayerNorm/gamma/read*'
_output_shapes
:џџџџџџџџџ*
T0
В
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumSumGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mulYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ь
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0A2S/best_value_network/LayerNorm/batchnorm/RsqrtA2S/gradients_1/AddN_1*'
_output_shapes
:џџџџџџџџџ*
T0
И
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/mul_1[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѓ
MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
њ
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1

\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:

OA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0A2S/best_value_network/LayerNorm/batchnorm/Rsqrt\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
К
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeShape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
YA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
К
GA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumSumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradYA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/SumIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
О
IA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumOA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad[A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

MA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeIA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Sum_1KA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
њ
TA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpL^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1

\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityKA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/ReshapeU^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityMA2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1U^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: *
T0
Ц
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeShape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:

KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addAddCA2S/best_value_network/LayerNorm/moments/variance/reduction_indicesKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modFloorModJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/addKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
ђ
LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeRangeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/startKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/SizeRA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0

QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/FillFillNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_1QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0
б
TA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/rangeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/modLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Fill*
N*#
_output_shapes
:џџџџџџџџџ*
T0

PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Џ
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/MaximumMaximumTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitchPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

OA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordivFloorDivLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ShapeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
О
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeReshape\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyTA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
С
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileTileNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ReshapeOA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0
Ш
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2Shape:A2S/best_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
П
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3Shape1A2S/best_value_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:

LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Џ
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_2LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Г
MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1ProdNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Shape_3NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1MaximumMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Prod_1RA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0

QA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/ProdPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
ж
KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/CastCastQA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Ѕ
NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truedivRealDivKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/TileKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_value_network/add*
_output_shapes
:*
T0*
out_type0
Ь
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5A2S/best_value_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
ы
eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ь
VA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarConstO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
Д
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulMulVA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/scalarNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
А
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_value_network/add5A2S/best_value_network/LayerNorm/moments/StopGradientO^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
И
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mulSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
и
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1eA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeSA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
м
UA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1gA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
д
YA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeUA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1WA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
ч
SA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegYA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0

`A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpX^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeT^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ю
hA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityWA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshapea^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*j
_class`
^\loc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ш
jA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitySA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Nega^A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
Ђ
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ќ
FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addAdd?A2S/best_value_network/LayerNorm/moments/mean/reduction_indicesGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

FA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modFloorModFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/addGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
т
HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeRangeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/startGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/SizeNA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0

MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0

GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/FillFillJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_1MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill/value*
T0*
_output_shapes
:
Н
PA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/rangeFA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/modHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ѓ
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/MaximumMaximumPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitchLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

KA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordivFloorDivHA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ShapeJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum*
T0*
_output_shapes
:
И
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeReshape^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyPA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
Е
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileTileJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ReshapeKA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Є
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_value_network/add*
T0*
out_type0*
_output_shapes
:
З
JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3Shape-A2S/best_value_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0

HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ѓ
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_2HA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ї
IA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1ProdJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Shape_3JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

LA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumIA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Prod_1NA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 

MA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/ProdLA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
Ю
GA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/CastCastMA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0

JA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truedivRealDivGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/TileGA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
б
A2S/gradients_1/AddN_2AddN^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyhA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/best_value_network/LayerNorm/moments/mean_grad/truediv*
T0*`
_classV
TRloc:@A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

5A2S/gradients_1/A2S/best_value_network/add_grad/ShapeShapeA2S/best_value_network/MatMul*
T0*
out_type0*
_output_shapes
:

7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs5A2S/gradients_1/A2S/best_value_network/add_grad/Shape7A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
й
3A2S/gradients_1/A2S/best_value_network/add_grad/SumSumA2S/gradients_1/AddN_2EA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeReshape3A2S/gradients_1/A2S/best_value_network/add_grad/Sum5A2S/gradients_1/A2S/best_value_network/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
н
5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_1SumA2S/gradients_1/AddN_2GA2S/gradients_1/A2S/best_value_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ч
9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1Reshape5A2S/gradients_1/A2S/best_value_network/add_grad/Sum_17A2S/gradients_1/A2S/best_value_network/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
О
@A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_depsNoOp8^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape:^A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1
Ю
HA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependencyIdentity7A2S/gradients_1/A2S/best_value_network/add_grad/ReshapeA^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ч
JA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1Identity9A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1A^A2S/gradients_1/A2S/best_value_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/add_grad/Reshape_1*
_output_shapes
:

9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulMatMulHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
№
;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsHA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
Х
CA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_depsNoOp:^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul<^A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1
и
KA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependencyIdentity9A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMulD^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
е
MA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1Identity;A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1D^A2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*N
_classD
B@loc:@A2S/gradients_1/A2S/best_value_network/MatMul_grad/MatMul_1

A2S/beta1_power_1/initial_valueConst*
valueB
 *fff?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Џ
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
к
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta

A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta

A2S/beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *wО?*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
dtype0
Џ
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
к
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 

A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
_output_shapes
: *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
у
JA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
№
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
х
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/w/AdamJA2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:
№
=A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
х
LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
ђ
:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
ы
AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
є
?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:
л
JA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
ш
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
с
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/fc0/b/AdamJA2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:
ь
=A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
н
LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB*    *
dtype0
ъ
:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:*
dtype0*
_output_shapes
:
ч
AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1LA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b
№
?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:
Ч
@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zerosConst*
_output_shapes
:*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0
д
.A2S/A2S/best_value_network/LayerNorm/beta/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
	container 
Й
5A2S/A2S/best_value_network/LayerNorm/beta/Adam/AssignAssign.A2S/A2S/best_value_network/LayerNorm/beta/Adam@A2S/A2S/best_value_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
Ю
3A2S/A2S/best_value_network/LayerNorm/beta/Adam/readIdentity.A2S/A2S/best_value_network/LayerNorm/beta/Adam*
_output_shapes
:*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta
Щ
BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
ж
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
П
7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/AssignAssign0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1BA2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
в
5A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/readIdentity0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
:
Щ
AA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
ж
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
Н
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/AssignAssign/A2S/A2S/best_value_network/LayerNorm/gamma/AdamAA2S/A2S/best_value_network/LayerNorm/gamma/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(
б
4A2S/A2S/best_value_network/LayerNorm/gamma/Adam/readIdentity/A2S/A2S/best_value_network/LayerNorm/gamma/Adam*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:*
T0
Ы
CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
и
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma
У
8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/AssignAssign1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1CA2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
е
6A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/readIdentity1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1*
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
_output_shapes
:
у
JA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
№
8A2S/A2S/best_value_network/best_value_network/out/w/Adam
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
х
?A2S/A2S/best_value_network/best_value_network/out/w/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/w/AdamJA2S/A2S/best_value_network/best_value_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:
№
=A2S/A2S/best_value_network/best_value_network/out/w/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/w/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:
х
LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB*    
ђ
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
ы
AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1LA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(
є
?A2S/A2S/best_value_network/best_value_network/out/w/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1*
_output_shapes

:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
л
JA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    
ш
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
с
?A2S/A2S/best_value_network/best_value_network/out/b/Adam/AssignAssign8A2S/A2S/best_value_network/best_value_network/out/b/AdamJA2S/A2S/best_value_network/best_value_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(
ь
=A2S/A2S/best_value_network/best_value_network/out/b/Adam/readIdentity8A2S/A2S/best_value_network/best_value_network/out/b/Adam*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
н
LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ъ
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
ч
AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAssign:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1LA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(
№
?A2S/A2S/best_value_network/best_value_network/out/b/Adam_1/readIdentity:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1*
_output_shapes
:*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b
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
 *wО?*
dtype0*
_output_shapes
: 
W
A2S/Adam_1/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
н
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/w8A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/best_value_network/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
use_nesterov( 
ж
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/fc0/b8A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonJA2S/gradients_1/A2S/best_value_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
use_nesterov( *
_output_shapes
:
Ж
AA2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam%A2S/best_value_network/LayerNorm/beta.A2S/A2S/best_value_network/LayerNorm/beta/Adam0A2S/A2S/best_value_network/LayerNorm/beta/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon\A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
Н
BA2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&A2S/best_value_network/LayerNorm/gamma/A2S/A2S/best_value_network/LayerNorm/gamma/Adam1A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilon^A2S/gradients_1/A2S/best_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*9
_class/
-+loc:@A2S/best_value_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
п
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/w8A2S/A2S/best_value_network/best_value_network/out/w/Adam:A2S/A2S/best_value_network/best_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/best_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
use_nesterov( *
_output_shapes

:
и
KA2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam	ApplyAdam/A2S/best_value_network/best_value_network/out/b8A2S/A2S/best_value_network/best_value_network/out/b/Adam:A2S/A2S/best_value_network/best_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonLA2S/gradients_1/A2S/best_value_network/add_2_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
л
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: 
Т
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
use_locking( *
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
н
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2L^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/fc0/b/ApplyAdamB^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/beta/ApplyAdamC^A2S/Adam_1/update_A2S/best_value_network/LayerNorm/gamma/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/w/ApplyAdamL^A2S/Adam_1/update_A2S/best_value_network/best_value_network/out/b/ApplyAdam*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
_output_shapes
: *
T0
Ц
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
T0*8
_class.
,*loc:@A2S/best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( 
§

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
 *  ?*
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
Ў
'A2S/gradients_2/A2S/Mean_4_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_4_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
|
%A2S/gradients_2/A2S/Mean_4_grad/ShapeShapeA2S/SquaredDifference_1*
_output_shapes
:*
T0*
out_type0
Р
$A2S/gradients_2/A2S/Mean_4_grad/TileTile'A2S/gradients_2/A2S/Mean_4_grad/Reshape%A2S/gradients_2/A2S/Mean_4_grad/Shape*
T0*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0
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
К
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
О
&A2S/gradients_2/A2S/Mean_4_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_4_grad/Shape_2'A2S/gradients_2/A2S/Mean_4_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)A2S/gradients_2/A2S/Mean_4_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
І
'A2S/gradients_2/A2S/Mean_4_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_4_grad/Prod_1)A2S/gradients_2/A2S/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0
Є
(A2S/gradients_2/A2S/Mean_4_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_4_grad/Prod'A2S/gradients_2/A2S/Mean_4_grad/Maximum*
T0*
_output_shapes
: 

$A2S/gradients_2/A2S/Mean_4_grad/CastCast(A2S/gradients_2/A2S/Mean_4_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_2/A2S/Mean_4_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_4_grad/Tile$A2S/gradients_2/A2S/Mean_4_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/best_q_network/add_2*
out_type0*
_output_shapes
:*
T0

4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1ShapeA2S/returns*
_output_shapes
:*
T0*
out_type0

BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ђ
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ч
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_4_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
К
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/best_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_4_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Я
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
я
0A2S/gradients_2/A2S/SquaredDifference_1_grad/SumSum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ѓ
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
0A2S/gradients_2/A2S/SquaredDifference_1_grad/NegNeg6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
=A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_depsNoOp5^A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape1^A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg
Т
EA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyIdentity4A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*G
_class=
;9loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape
М
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg*'
_output_shapes
:џџџџџџџџџ

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

CA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

1A2S/gradients_2/A2S/best_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
5A2S/gradients_2/A2S/best_q_network/add_2_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
с
7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_2_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
И
>A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1
Ц
FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
П
HA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_2_grad/Reshape_1

7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulMatMulFA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
є
9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1MatMulA2S/best_q_network/add_1FA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
П
AA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_depsNoOp8^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul:^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1
а
IA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyIdentity7A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMulB^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Э
KA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1Identity9A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1B^A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*L
_classB
@>loc:@A2S/gradients_2/A2S/best_q_network/MatMul_1_grad/MatMul_1

3A2S/gradients_2/A2S/best_q_network/add_1_grad/ShapeShapeA2S/best_q_network/mul*
T0*
out_type0*
_output_shapes
:

5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1ShapeA2S/best_q_network/mul_1*
out_type0*
_output_shapes
:*
T0

CA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

1A2S/gradients_2/A2S/best_q_network/add_1_grad/SumSumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyCA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ш
5A2S/gradients_2/A2S/best_q_network/add_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_1SumIA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependencyEA2S/gradients_2/A2S/best_q_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/add_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/add_1_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
И
>A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1
Ц
FA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ь
HA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
t
1A2S/gradients_2/A2S/best_q_network/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
џ
AA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape3A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
/A2S/gradients_2/A2S/best_q_network/mul_grad/mulMulFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency,A2S/best_q_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/A2S/gradients_2/A2S/best_q_network/mul_grad/SumSum/A2S/gradients_2/A2S/best_q_network/mul_grad/mulAA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
3A2S/gradients_2/A2S/best_q_network/mul_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/mul_grad/Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ь
1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1MulA2S/best_q_network/mul/xFA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
№
1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_1Sum1A2S/gradients_2/A2S/best_q_network/mul_grad/mul_1CA2S/gradients_2/A2S/best_q_network/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ш
5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/mul_grad/Sum_13A2S/gradients_2/A2S/best_q_network/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
В
<A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1
­
DA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape*
_output_shapes
: 
Ф
FA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
v
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1ShapeA2S/best_q_network/Abs*
T0*
out_type0*
_output_shapes
:

CA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ь
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulMulHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1A2S/best_q_network/Abs*
T0*'
_output_shapes
:џџџџџџџџџ
№
1A2S/gradients_2/A2S/best_q_network/mul_1_grad/SumSum1A2S/gradients_2/A2S/best_q_network/mul_1_grad/mulCA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
з
5A2S/gradients_2/A2S/best_q_network/mul_1_grad/ReshapeReshape1A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
в
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1MulA2S/best_q_network/mul_1/xHA2S/gradients_2/A2S/best_q_network/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
і
3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_1Sum3A2S/gradients_2/A2S/best_q_network/mul_1_grad/mul_1EA2S/gradients_2/A2S/best_q_network/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ю
7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1Reshape3A2S/gradients_2/A2S/best_q_network/mul_1_grad/Sum_15A2S/gradients_2/A2S/best_q_network/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
И
>A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape8^A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1
Е
FA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape
Ь
HA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1?^A2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/group_deps*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0

0A2S/gradients_2/A2S/best_q_network/Abs_grad/SignSign,A2S/best_q_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
ф
/A2S/gradients_2/A2S/best_q_network/Abs_grad/mulMulHA2S/gradients_2/A2S/best_q_network/mul_1_grad/tuple/control_dependency_10A2S/gradients_2/A2S/best_q_network/Abs_grad/Sign*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/gradients_2/AddNAddNFA2S/gradients_2/A2S/best_q_network/mul_grad/tuple/control_dependency_1/A2S/gradients_2/A2S/best_q_network/Abs_grad/mul*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/mul_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Г
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
Г
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
С
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ћ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumSumA2S/gradients_2/AddNWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Є
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
џ
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1SumA2S/gradients_2/AddNYA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapeA2S/best_q_network/add*
_output_shapes
:*
T0*
out_type0
Г
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
С
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*A2S/best_q_network/LayerNorm/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Є
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
є
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1MulA2S/best_q_network/addZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
В
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Г
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape,A2S/best_q_network/LayerNorm/batchnorm/mul_2*
_output_shapes
:*
T0*
out_type0
Л
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
П
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumSum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
У
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Д
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegNegEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
Ђ
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/NegGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ю
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1

XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
А
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape)A2S/best_q_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
Г
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
С
WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*A2S/best_q_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumSumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mulWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Є
IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul)A2S/best_q_network/LayerNorm/moments/meanZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
В
GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1YA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Њ
KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
є
RA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeS^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
ѕ
A2S/gradients_2/AddN_1AddN\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:џџџџџџџџџ
Б
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeShape,A2S/best_q_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulMulA2S/gradients_2/AddN_1'A2S/best_q_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:џџџџџџџџџ
І
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumSumCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mulUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ф
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul,A2S/best_q_network/LayerNorm/batchnorm/RsqrtA2S/gradients_2/AddN_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/mul_1WA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
ю
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1

XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:

KA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad,A2S/best_q_network/LayerNorm/batchnorm/RsqrtXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
В
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeShape-A2S/best_q_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Л
UA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ў
CA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumSumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/SumEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
В
EA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1SumKA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

IA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeEA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Sum_1GA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ю
PA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpH^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeJ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1

XA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/ReshapeQ^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

ZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/Reshape_1
О
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeShape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:

GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ќ
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addAdd?A2S/best_q_network/LayerNorm/moments/variance/reduction_indicesGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modFloorModFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/addGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Size*
T0*
_output_shapes
:

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
т
HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeRangeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/startGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/SizeNA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0

MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0

GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/FillFillJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_1MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0
Н
PA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/rangeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/modHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ѓ
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/MaximumMaximumPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitchLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0

KA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordivFloorDivHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ShapeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum*
T0*
_output_shapes
:
В
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeReshapeXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyPA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Е
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileTileJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ReshapeKA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2Shape6A2S/best_q_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
З
JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3Shape-A2S/best_q_network/LayerNorm/moments/variance*
out_type0*
_output_shapes
:*
T0

HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ѓ
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_2HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ї
IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1ProdJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Shape_3JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Prod_1NA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*
_output_shapes
: 

MA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/ProdLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0
Ю
GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/CastCastMA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truedivRealDivGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/TileGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapeA2S/best_q_network/add*
out_type0*
_output_shapes
:*
T0
Ф
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape1A2S/best_q_network/LayerNorm/moments/StopGradient*
out_type0*
_output_shapes
:*
T0
п
aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeSA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ф
RA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
Ј
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulMulRA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/scalarJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
 
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/subSubA2S/best_q_network/add1A2S/best_q_network/LayerNorm/moments/StopGradientK^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mulOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1aA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Т
SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
а
QA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1cA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ш
UA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
п
OA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/NegNegUA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0

\A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpT^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeP^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
О
dA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*f
_class\
ZXloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape
И
fA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg]^A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*b
_classX
VTloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ

DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeShapeA2S/best_q_network/add*
T0*
out_type0*
_output_shapes
:

CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
№
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addAdd;A2S/best_q_network/LayerNorm/moments/mean/reduction_indicesCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:
ќ
BA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modFloorModBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/addCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Size*
T0*
_output_shapes
:

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
в
DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeRangeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/startCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/SizeJA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0

IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/FillFillFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_1IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0
Љ
LA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/rangeBA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/modDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ

HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/MaximumMaximumLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitchHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ

GA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordivFloorDivDA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ShapeFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0
Ќ
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Љ
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileTileFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ReshapeGA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2ShapeA2S/best_q_network/add*
out_type0*
_output_shapes
:*
T0
Џ
FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3Shape)A2S/best_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0

DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_2DA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

EA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1ProdFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Shape_3FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

HA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Prod_1JA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*
_output_shapes
: 

IA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/ProdHA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0
Ц
CA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/CastCastIA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0

FA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truedivRealDivCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/TileCA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
С
A2S/gradients_2/AddN_2AddNZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFA2S/gradients_2/A2S/best_q_network/LayerNorm/moments/mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0*\
_classR
PNloc:@A2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N

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
џ
AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1A2S/gradients_2/A2S/best_q_network/add_grad/Shape3A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
б
/A2S/gradients_2/A2S/best_q_network/add_grad/SumSumA2S/gradients_2/AddN_2AA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
т
3A2S/gradients_2/A2S/best_q_network/add_grad/ReshapeReshape/A2S/gradients_2/A2S/best_q_network/add_grad/Sum1A2S/gradients_2/A2S/best_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
е
1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_1SumA2S/gradients_2/AddN_2CA2S/gradients_2/A2S/best_q_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
л
5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1Reshape1A2S/gradients_2/A2S/best_q_network/add_grad/Sum_13A2S/gradients_2/A2S/best_q_network/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
В
<A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_depsNoOp4^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape6^A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1
О
DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependencyIdentity3A2S/gradients_2/A2S/best_q_network/add_grad/Reshape=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
З
FA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1Identity5A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1=^A2S/gradients_2/A2S/best_q_network/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/add_grad/Reshape_1*
_output_shapes
:

5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMulMatMulDA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ф
7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1DA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Й
?A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_depsNoOp6^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul8^A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1
Ш
GA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependencyIdentity5A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*H
_class>
<:loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul
Х
IA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1Identity7A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1@^A2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/group_deps*J
_class@
><loc:@A2S/gradients_2/A2S/best_q_network/MatMul_grad/MatMul_1*
_output_shapes

:*
T0

A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Ћ
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
ж
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
_output_shapes
: *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta

A2S/beta2_power_2/initial_valueConst*
valueB
 *wО?*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
Ћ
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
ж
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 

A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
г
BA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
р
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
Х
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/w/AdamBA2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Initializer/zeros*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
и
5A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:
е
DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB*    *
dtype0*
_output_shapes

:
т
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
Ы
9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:
м
7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1*
_output_shapes

:*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
Ы
BA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
и
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
С
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/fc0/b/AdamBA2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:*
use_locking(
д
5A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
Э
DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB*    *
dtype0*
_output_shapes
:
к
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
Ч
9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1DA2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:
и
7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:
П
<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Ь
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
Љ
1A2S/A2S/best_q_network/LayerNorm/beta/Adam/AssignAssign*A2S/A2S/best_q_network/LayerNorm/beta/Adam<A2S/A2S/best_q_network/LayerNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:
Т
/A2S/A2S/best_q_network/LayerNorm/beta/Adam/readIdentity*A2S/A2S/best_q_network/LayerNorm/beta/Adam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
:
С
>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
Ю
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
Џ
3A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/AssignAssign,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1>A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Initializer/zeros*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ц
1A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/readIdentity,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1*
_output_shapes
:*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
С
=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
Ю
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
­
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/AssignAssign+A2S/A2S/best_q_network/LayerNorm/gamma/Adam=A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Initializer/zeros*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Х
0A2S/A2S/best_q_network/LayerNorm/gamma/Adam/readIdentity+A2S/A2S/best_q_network/LayerNorm/gamma/Adam*
_output_shapes
:*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma
У
?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
valueB*    *
dtype0*
_output_shapes
:
а
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
Г
4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/AssignAssign-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1?A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:
Щ
2A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/readIdentity-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1*
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
_output_shapes
:
г
BA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    
р
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
Х
7A2S/A2S/best_q_network/best_q_network/out/w/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/w/AdamBA2S/A2S/best_q_network/best_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
и
5A2S/A2S/best_q_network/best_q_network/out/w/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/w/Adam*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
е
DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB*    *
dtype0*
_output_shapes

:
т
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
Ы
9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1DA2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:
м
7A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/readIdentity2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:
Ы
BA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zerosConst*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0
и
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
С
7A2S/A2S/best_q_network/best_q_network/out/b/Adam/AssignAssign0A2S/A2S/best_q_network/best_q_network/out/b/AdamBA2S/A2S/best_q_network/best_q_network/out/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
д
5A2S/A2S/best_q_network/best_q_network/out/b/Adam/readIdentity0A2S/A2S/best_q_network/best_q_network/out/b/Adam*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:*
T0
Э
DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
к
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
Ч
9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/AssignAssign2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1DA2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
и
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
 *wО?*
dtype0*
_output_shapes
: 
W
A2S/Adam_2/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Б
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/w0A2S/A2S/best_q_network/best_q_network/fc0/w/Adam2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/best_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
use_nesterov( *
_output_shapes

:
Њ
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/fc0/b0A2S/A2S/best_q_network/best_q_network/fc0/b/Adam2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonFA2S/gradients_2/A2S/best_q_network/add_grad/tuple/control_dependency_1*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
use_nesterov( *
_output_shapes
:*
use_locking( 

=A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam!A2S/best_q_network/LayerNorm/beta*A2S/A2S/best_q_network/LayerNorm/beta/Adam,A2S/A2S/best_q_network/LayerNorm/beta/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonXA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:
Ѕ
>A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam"A2S/best_q_network/LayerNorm/gamma+A2S/A2S/best_q_network/LayerNorm/gamma/Adam-A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonZA2S/gradients_2/A2S/best_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*5
_class+
)'loc:@A2S/best_q_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:
Г
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/w0A2S/A2S/best_q_network/best_q_network/out/w/Adam2A2S/A2S/best_q_network/best_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/best_q_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
use_nesterov( *
_output_shapes

:
Ќ
CA2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam	ApplyAdam'A2S/best_q_network/best_q_network/out/b0A2S/A2S/best_q_network/best_q_network/out/b/Adam2A2S/A2S/best_q_network/best_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonHA2S/gradients_2/A2S/best_q_network/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
use_nesterov( 
Џ
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
_output_shapes
: 
О
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(
Б
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2D^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam*
_output_shapes
: *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta
Т
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
use_locking( *
T0*4
_class*
(&loc:@A2S/best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
е

A2S/Adam_2NoOpD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/fc0/b/ApplyAdam>^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/beta/ApplyAdam?^A2S/Adam_2/update_A2S/best_q_network/LayerNorm/gamma/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/w/ApplyAdamD^A2S/Adam_2/update_A2S/best_q_network/best_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1

A2S/group_depsNoOp

A2S/group_deps_1NoOp
Ћ
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
A2S/average_reward_1ScalarSummaryA2S/average_reward_1/tagsA2S/average_reward*
_output_shapes
: *
T0""
model_variablesђя

*A2S/backup_policy_network/LayerNorm/beta:0/A2S/backup_policy_network/LayerNorm/beta/Assign/A2S/backup_policy_network/LayerNorm/beta/read:0

+A2S/backup_policy_network/LayerNorm/gamma:00A2S/backup_policy_network/LayerNorm/gamma/Assign0A2S/backup_policy_network/LayerNorm/gamma/read:0

(A2S/best_policy_network/LayerNorm/beta:0-A2S/best_policy_network/LayerNorm/beta/Assign-A2S/best_policy_network/LayerNorm/beta/read:0

)A2S/best_policy_network/LayerNorm/gamma:0.A2S/best_policy_network/LayerNorm/gamma/Assign.A2S/best_policy_network/LayerNorm/gamma/read:0

)A2S/backup_value_network/LayerNorm/beta:0.A2S/backup_value_network/LayerNorm/beta/Assign.A2S/backup_value_network/LayerNorm/beta/read:0

*A2S/backup_value_network/LayerNorm/gamma:0/A2S/backup_value_network/LayerNorm/gamma/Assign/A2S/backup_value_network/LayerNorm/gamma/read:0

'A2S/best_value_network/LayerNorm/beta:0,A2S/best_value_network/LayerNorm/beta/Assign,A2S/best_value_network/LayerNorm/beta/read:0

(A2S/best_value_network/LayerNorm/gamma:0-A2S/best_value_network/LayerNorm/gamma/Assign-A2S/best_value_network/LayerNorm/gamma/read:0

%A2S/backup_q_network/LayerNorm/beta:0*A2S/backup_q_network/LayerNorm/beta/Assign*A2S/backup_q_network/LayerNorm/beta/read:0

&A2S/backup_q_network/LayerNorm/gamma:0+A2S/backup_q_network/LayerNorm/gamma/Assign+A2S/backup_q_network/LayerNorm/gamma/read:0
y
#A2S/best_q_network/LayerNorm/beta:0(A2S/best_q_network/LayerNorm/beta/Assign(A2S/best_q_network/LayerNorm/beta/read:0
|
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0"
	summaries

A2S/kl:0
A2S/average_advantage:0
A2S/policy_network_loss:0
A2S/value_network_loss:0
A2S/q_network_loss:0
A2S/average_reward_1:0",
trainable_variables,џ+
Е
7A2S/backup_policy_network/backup_policy_network/fc0/w:0<A2S/backup_policy_network/backup_policy_network/fc0/w/Assign<A2S/backup_policy_network/backup_policy_network/fc0/w/read:0
Е
7A2S/backup_policy_network/backup_policy_network/fc0/b:0<A2S/backup_policy_network/backup_policy_network/fc0/b/Assign<A2S/backup_policy_network/backup_policy_network/fc0/b/read:0

*A2S/backup_policy_network/LayerNorm/beta:0/A2S/backup_policy_network/LayerNorm/beta/Assign/A2S/backup_policy_network/LayerNorm/beta/read:0

+A2S/backup_policy_network/LayerNorm/gamma:00A2S/backup_policy_network/LayerNorm/gamma/Assign0A2S/backup_policy_network/LayerNorm/gamma/read:0
Е
7A2S/backup_policy_network/backup_policy_network/out/w:0<A2S/backup_policy_network/backup_policy_network/out/w/Assign<A2S/backup_policy_network/backup_policy_network/out/w/read:0
Е
7A2S/backup_policy_network/backup_policy_network/out/b:0<A2S/backup_policy_network/backup_policy_network/out/b/Assign<A2S/backup_policy_network/backup_policy_network/out/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0

(A2S/best_policy_network/LayerNorm/beta:0-A2S/best_policy_network/LayerNorm/beta/Assign-A2S/best_policy_network/LayerNorm/beta/read:0

)A2S/best_policy_network/LayerNorm/gamma:0.A2S/best_policy_network/LayerNorm/gamma/Assign.A2S/best_policy_network/LayerNorm/gamma/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
Џ
5A2S/backup_value_network/backup_value_network/fc0/w:0:A2S/backup_value_network/backup_value_network/fc0/w/Assign:A2S/backup_value_network/backup_value_network/fc0/w/read:0
Џ
5A2S/backup_value_network/backup_value_network/fc0/b:0:A2S/backup_value_network/backup_value_network/fc0/b/Assign:A2S/backup_value_network/backup_value_network/fc0/b/read:0

)A2S/backup_value_network/LayerNorm/beta:0.A2S/backup_value_network/LayerNorm/beta/Assign.A2S/backup_value_network/LayerNorm/beta/read:0

*A2S/backup_value_network/LayerNorm/gamma:0/A2S/backup_value_network/LayerNorm/gamma/Assign/A2S/backup_value_network/LayerNorm/gamma/read:0
Џ
5A2S/backup_value_network/backup_value_network/out/w:0:A2S/backup_value_network/backup_value_network/out/w/Assign:A2S/backup_value_network/backup_value_network/out/w/read:0
Џ
5A2S/backup_value_network/backup_value_network/out/b:0:A2S/backup_value_network/backup_value_network/out/b/Assign:A2S/backup_value_network/backup_value_network/out/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0

'A2S/best_value_network/LayerNorm/beta:0,A2S/best_value_network/LayerNorm/beta/Assign,A2S/best_value_network/LayerNorm/beta/read:0

(A2S/best_value_network/LayerNorm/gamma:0-A2S/best_value_network/LayerNorm/gamma/Assign-A2S/best_value_network/LayerNorm/gamma/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0

-A2S/backup_q_network/backup_q_network/fc0/w:02A2S/backup_q_network/backup_q_network/fc0/w/Assign2A2S/backup_q_network/backup_q_network/fc0/w/read:0

-A2S/backup_q_network/backup_q_network/fc0/b:02A2S/backup_q_network/backup_q_network/fc0/b/Assign2A2S/backup_q_network/backup_q_network/fc0/b/read:0

%A2S/backup_q_network/LayerNorm/beta:0*A2S/backup_q_network/LayerNorm/beta/Assign*A2S/backup_q_network/LayerNorm/beta/read:0

&A2S/backup_q_network/LayerNorm/gamma:0+A2S/backup_q_network/LayerNorm/gamma/Assign+A2S/backup_q_network/LayerNorm/gamma/read:0

-A2S/backup_q_network/backup_q_network/out/w:02A2S/backup_q_network/backup_q_network/out/w/Assign2A2S/backup_q_network/backup_q_network/out/w/read:0

-A2S/backup_q_network/backup_q_network/out/b:02A2S/backup_q_network/backup_q_network/out/b/Assign2A2S/backup_q_network/backup_q_network/out/b/read:0

)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0

)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0
y
#A2S/best_q_network/LayerNorm/beta:0(A2S/best_q_network/LayerNorm/beta/Assign(A2S/best_q_network/LayerNorm/beta/read:0
|
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0

)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0

)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0"0
train_op$
"
A2S/Adam

A2S/Adam_1

A2S/Adam_2"Ьb
	variablesОbЛb
Е
7A2S/backup_policy_network/backup_policy_network/fc0/w:0<A2S/backup_policy_network/backup_policy_network/fc0/w/Assign<A2S/backup_policy_network/backup_policy_network/fc0/w/read:0
Е
7A2S/backup_policy_network/backup_policy_network/fc0/b:0<A2S/backup_policy_network/backup_policy_network/fc0/b/Assign<A2S/backup_policy_network/backup_policy_network/fc0/b/read:0

*A2S/backup_policy_network/LayerNorm/beta:0/A2S/backup_policy_network/LayerNorm/beta/Assign/A2S/backup_policy_network/LayerNorm/beta/read:0

+A2S/backup_policy_network/LayerNorm/gamma:00A2S/backup_policy_network/LayerNorm/gamma/Assign0A2S/backup_policy_network/LayerNorm/gamma/read:0
Е
7A2S/backup_policy_network/backup_policy_network/out/w:0<A2S/backup_policy_network/backup_policy_network/out/w/Assign<A2S/backup_policy_network/backup_policy_network/out/w/read:0
Е
7A2S/backup_policy_network/backup_policy_network/out/b:0<A2S/backup_policy_network/backup_policy_network/out/b/Assign<A2S/backup_policy_network/backup_policy_network/out/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0

(A2S/best_policy_network/LayerNorm/beta:0-A2S/best_policy_network/LayerNorm/beta/Assign-A2S/best_policy_network/LayerNorm/beta/read:0

)A2S/best_policy_network/LayerNorm/gamma:0.A2S/best_policy_network/LayerNorm/gamma/Assign.A2S/best_policy_network/LayerNorm/gamma/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
Џ
5A2S/backup_value_network/backup_value_network/fc0/w:0:A2S/backup_value_network/backup_value_network/fc0/w/Assign:A2S/backup_value_network/backup_value_network/fc0/w/read:0
Џ
5A2S/backup_value_network/backup_value_network/fc0/b:0:A2S/backup_value_network/backup_value_network/fc0/b/Assign:A2S/backup_value_network/backup_value_network/fc0/b/read:0

)A2S/backup_value_network/LayerNorm/beta:0.A2S/backup_value_network/LayerNorm/beta/Assign.A2S/backup_value_network/LayerNorm/beta/read:0

*A2S/backup_value_network/LayerNorm/gamma:0/A2S/backup_value_network/LayerNorm/gamma/Assign/A2S/backup_value_network/LayerNorm/gamma/read:0
Џ
5A2S/backup_value_network/backup_value_network/out/w:0:A2S/backup_value_network/backup_value_network/out/w/Assign:A2S/backup_value_network/backup_value_network/out/w/read:0
Џ
5A2S/backup_value_network/backup_value_network/out/b:0:A2S/backup_value_network/backup_value_network/out/b/Assign:A2S/backup_value_network/backup_value_network/out/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0

'A2S/best_value_network/LayerNorm/beta:0,A2S/best_value_network/LayerNorm/beta/Assign,A2S/best_value_network/LayerNorm/beta/read:0

(A2S/best_value_network/LayerNorm/gamma:0-A2S/best_value_network/LayerNorm/gamma/Assign-A2S/best_value_network/LayerNorm/gamma/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0

-A2S/backup_q_network/backup_q_network/fc0/w:02A2S/backup_q_network/backup_q_network/fc0/w/Assign2A2S/backup_q_network/backup_q_network/fc0/w/read:0

-A2S/backup_q_network/backup_q_network/fc0/b:02A2S/backup_q_network/backup_q_network/fc0/b/Assign2A2S/backup_q_network/backup_q_network/fc0/b/read:0

%A2S/backup_q_network/LayerNorm/beta:0*A2S/backup_q_network/LayerNorm/beta/Assign*A2S/backup_q_network/LayerNorm/beta/read:0

&A2S/backup_q_network/LayerNorm/gamma:0+A2S/backup_q_network/LayerNorm/gamma/Assign+A2S/backup_q_network/LayerNorm/gamma/read:0

-A2S/backup_q_network/backup_q_network/out/w:02A2S/backup_q_network/backup_q_network/out/w/Assign2A2S/backup_q_network/backup_q_network/out/w/read:0

-A2S/backup_q_network/backup_q_network/out/b:02A2S/backup_q_network/backup_q_network/out/b/Assign2A2S/backup_q_network/backup_q_network/out/b/read:0

)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0

)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0
y
#A2S/best_q_network/LayerNorm/beta:0(A2S/best_q_network/LayerNorm/beta/Assign(A2S/best_q_network/LayerNorm/beta/read:0
|
$A2S/best_q_network/LayerNorm/gamma:0)A2S/best_q_network/LayerNorm/gamma/Assign)A2S/best_q_network/LayerNorm/gamma/read:0

)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0

)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0
C
A2S/beta1_power:0A2S/beta1_power/AssignA2S/beta1_power/read:0
C
A2S/beta2_power:0A2S/beta2_power/AssignA2S/beta2_power/read:0
Ф
<A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam:0AA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam/read:0
Ъ
>A2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/fc0/w/Adam_1/read:0
Ф
<A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam:0AA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam/read:0
Ъ
>A2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/fc0/b/Adam_1/read:0
Ѓ
1A2S/A2S/best_policy_network/LayerNorm/beta/Adam:06A2S/A2S/best_policy_network/LayerNorm/beta/Adam/Assign6A2S/A2S/best_policy_network/LayerNorm/beta/Adam/read:0
Љ
3A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1:08A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/Assign8A2S/A2S/best_policy_network/LayerNorm/beta/Adam_1/read:0
І
2A2S/A2S/best_policy_network/LayerNorm/gamma/Adam:07A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/Assign7A2S/A2S/best_policy_network/LayerNorm/gamma/Adam/read:0
Ќ
4A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1:09A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/Assign9A2S/A2S/best_policy_network/LayerNorm/gamma/Adam_1/read:0
Ф
<A2S/A2S/best_policy_network/best_policy_network/out/w/Adam:0AA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/out/w/Adam/read:0
Ъ
>A2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/out/w/Adam_1/read:0
Ф
<A2S/A2S/best_policy_network/best_policy_network/out/b/Adam:0AA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/AssignAA2S/A2S/best_policy_network/best_policy_network/out/b/Adam/read:0
Ъ
>A2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1:0CA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/AssignCA2S/A2S/best_policy_network/best_policy_network/out/b/Adam_1/read:0
I
A2S/beta1_power_1:0A2S/beta1_power_1/AssignA2S/beta1_power_1/read:0
I
A2S/beta2_power_1:0A2S/beta2_power_1/AssignA2S/beta2_power_1/read:0
О
:A2S/A2S/best_value_network/best_value_network/fc0/w/Adam:0?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/Assign?A2S/A2S/best_value_network/best_value_network/fc0/w/Adam/read:0
Ф
<A2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1:0AA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/fc0/w/Adam_1/read:0
О
:A2S/A2S/best_value_network/best_value_network/fc0/b/Adam:0?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/Assign?A2S/A2S/best_value_network/best_value_network/fc0/b/Adam/read:0
Ф
<A2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1:0AA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/fc0/b/Adam_1/read:0
 
0A2S/A2S/best_value_network/LayerNorm/beta/Adam:05A2S/A2S/best_value_network/LayerNorm/beta/Adam/Assign5A2S/A2S/best_value_network/LayerNorm/beta/Adam/read:0
І
2A2S/A2S/best_value_network/LayerNorm/beta/Adam_1:07A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/Assign7A2S/A2S/best_value_network/LayerNorm/beta/Adam_1/read:0
Ѓ
1A2S/A2S/best_value_network/LayerNorm/gamma/Adam:06A2S/A2S/best_value_network/LayerNorm/gamma/Adam/Assign6A2S/A2S/best_value_network/LayerNorm/gamma/Adam/read:0
Љ
3A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1:08A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/Assign8A2S/A2S/best_value_network/LayerNorm/gamma/Adam_1/read:0
О
:A2S/A2S/best_value_network/best_value_network/out/w/Adam:0?A2S/A2S/best_value_network/best_value_network/out/w/Adam/Assign?A2S/A2S/best_value_network/best_value_network/out/w/Adam/read:0
Ф
<A2S/A2S/best_value_network/best_value_network/out/w/Adam_1:0AA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/out/w/Adam_1/read:0
О
:A2S/A2S/best_value_network/best_value_network/out/b/Adam:0?A2S/A2S/best_value_network/best_value_network/out/b/Adam/Assign?A2S/A2S/best_value_network/best_value_network/out/b/Adam/read:0
Ф
<A2S/A2S/best_value_network/best_value_network/out/b/Adam_1:0AA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/AssignAA2S/A2S/best_value_network/best_value_network/out/b/Adam_1/read:0
I
A2S/beta1_power_2:0A2S/beta1_power_2/AssignA2S/beta1_power_2/read:0
I
A2S/beta2_power_2:0A2S/beta2_power_2/AssignA2S/beta2_power_2/read:0
І
2A2S/A2S/best_q_network/best_q_network/fc0/w/Adam:07A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/Assign7A2S/A2S/best_q_network/best_q_network/fc0/w/Adam/read:0
Ќ
4A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1:09A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/fc0/w/Adam_1/read:0
І
2A2S/A2S/best_q_network/best_q_network/fc0/b/Adam:07A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/Assign7A2S/A2S/best_q_network/best_q_network/fc0/b/Adam/read:0
Ќ
4A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1:09A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/fc0/b/Adam_1/read:0

,A2S/A2S/best_q_network/LayerNorm/beta/Adam:01A2S/A2S/best_q_network/LayerNorm/beta/Adam/Assign1A2S/A2S/best_q_network/LayerNorm/beta/Adam/read:0

.A2S/A2S/best_q_network/LayerNorm/beta/Adam_1:03A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/Assign3A2S/A2S/best_q_network/LayerNorm/beta/Adam_1/read:0

-A2S/A2S/best_q_network/LayerNorm/gamma/Adam:02A2S/A2S/best_q_network/LayerNorm/gamma/Adam/Assign2A2S/A2S/best_q_network/LayerNorm/gamma/Adam/read:0

/A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1:04A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/Assign4A2S/A2S/best_q_network/LayerNorm/gamma/Adam_1/read:0
І
2A2S/A2S/best_q_network/best_q_network/out/w/Adam:07A2S/A2S/best_q_network/best_q_network/out/w/Adam/Assign7A2S/A2S/best_q_network/best_q_network/out/w/Adam/read:0
Ќ
4A2S/A2S/best_q_network/best_q_network/out/w/Adam_1:09A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/out/w/Adam_1/read:0
І
2A2S/A2S/best_q_network/best_q_network/out/b/Adam:07A2S/A2S/best_q_network/best_q_network/out/b/Adam/Assign7A2S/A2S/best_q_network/best_q_network/out/b/Adam/read:0
Ќ
4A2S/A2S/best_q_network/best_q_network/out/b/Adam_1:09A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/Assign9A2S/A2S/best_q_network/best_q_network/out/b/Adam_1/read:0N`(       џpJ	eъПzжA*

A2S/average_reward_1  AУЧ(       џpJ	vяПzжA*

A2S/average_reward_1  Az*(       џpJ	8єПzжA*

A2S/average_reward_1ЋЊ
AтGЬ(       џpJ	ТЮјПzжA*

A2S/average_reward_1  AёsЧ$(       џpJ	ДўПzжA*

A2S/average_reward_133AќнRP(       џpJ	кПzжA*

A2S/average_reward_1UUAщФ
(       џpJ	5
ПzжA*

A2S/average_reward_1ЗmAуЩ(       џpJ	ГьПzжA*

A2S/average_reward_1  AІ(       џpJ	lПzжA*

A2S/average_reward_1ЧqAёЛJ(       џpJ	ПzжA*

A2S/average_reward_1ЭЬA"Ў2^(       џpJ	Г ПzжA*

A2S/average_reward_1ЃAЃЎMФ(       џpJ	'ПzжA*

A2S/average_reward_1UU!A*s(       џpJ	.ПzжA*

A2S/average_reward_1Oь$ADЖ=(       џpJ	ьl4ПzжA*

A2S/average_reward_1I$Aм(       џpJ	7ђ;ПzжA*

A2S/average_reward_1ww'ALХVБ(       џpJ	УBПzжA*

A2S/average_reward_1  )AОz	(       џpJ	GkGПzжA*

A2S/average_reward_1ZZ*A:§hа(       џpJ	лLПzжA*

A2S/average_reward_19+Aѕѓaф(       џpJ	PRПzжA*

A2S/average_reward_1ЏЁ,A+NЏ(       џpJ	дЉUПzжA*

A2S/average_reward_1ff*AZ
(       џpJ	нЌ[ПzжA*

A2S/average_reward_1У0,A!Ѓ(       џpJ	8FaПzжA*

A2S/average_reward_1щЂ+AѕыЗK(       џpJ	sхgПzжA*

A2S/average_reward_1Ї7-A>GWш(       џpJ	#ЊmПzжA*

A2S/average_reward_1ЋЊ,A&+y(       џpJ	ёqПzжA*

A2S/average_reward_1q=*AP:Ka(       џpJ	шuwПzжA*

A2S/average_reward_1;)A'Чk(       џpJ	й4ПzжA*

A2S/average_reward_1ЋЊ*Ayаo(       џpJ	N7ПzжA*

A2S/average_reward_1  ,AЇХч(       џpJ	АПzжA*

A2S/average_reward_1Xю)A`Жј(       џpJ	/ПzжA*

A2S/average_reward_1ЋЊ*AзgMг(       џpJ	юGПzжA*

A2S/average_reward_1R*Akс(       џpJ	фПzжA*

A2S/average_reward_1 *AѕwkР(       џpJ	ЎљЁПzжA*

A2S/average_reward_1щЂ+AЋ4*ш(       џpJ	ЈПzжA*

A2S/average_reward_1ЕД,ADcЛC(       џpJ	\­ПzжA*

A2S/average_reward_1От+ADМы:(       џpJ	;lГПzжA*

A2S/average_reward_19+AэQЎ(       џpJ	ж(ЛПzжA*

A2S/average_reward_1љ,AoтЄ(       џpJ	JhТПzжA*

A2S/average_reward_1Cy-AЮш(       џpJ	mтЪПzжA*

A2S/average_reward_1ц[.AШв(       џpJ	бПzжA*

A2S/average_reward_1ff.A&ж(       џpJ	њЭзПzжA*

A2S/average_reward_1dp.A-6п(       џpJ	жъмПzжA*

A2S/average_reward_1b.A=э5(       џpJ	dуПzжA*

A2S/average_reward_1<т.A}М(       џpJ	ЅЎшПzжA*

A2S/average_reward_1щЂ/Au
В(       џpJ	њKьПzжA*

A2S/average_reward_1щ.A2NЯ­(       џpJ	ѓПzжA*

A2S/average_reward_1ъM/A§ЅE(       џpJ	ИђјПzжA*

A2S/average_reward_1њ.Aіl(       џpJ	Aы§ПzжA*

A2S/average_reward_1UU.AH&ЯН(       џpJ	ЩПzжA*

A2S/average_reward_19/Asь(       џpJ	кПzжA*

A2S/average_reward_1  0A=тш(       џpJ	иПzжA*

A2S/average_reward_1ПО.AQe(       џpJ	aПzжA*

A2S/average_reward_1;Б/Aфt(       џpJ	bэ"ПzжA*

A2S/average_reward_1HM0AњЈ(       џpJ	т&ПzжA*

A2S/average_reward_1Lh/AX!N#(       џpJ	;,ПzжA*

A2S/average_reward_1Е/AЊ=^(       џpJ	b0ПzжA*

A2S/average_reward_1I.AјR}e(       џpJ	x4ПzжA*

A2S/average_reward_1ќ.AђІўP(       џpJ	Аm9ПzжA*

A2S/average_reward_1Ј.AЌ	(       џpJ	жВ=ПzжA*

A2S/average_reward_10-AЕл(       џpJ	гaCПzжA*

A2S/average_reward_1"".AпЁ*       L	vM		жЪПzжA*

A2S/kl:Аь<

A2S/average_advantageНлюН

A2S/policy_network_losss"ЖО

A2S/value_network_loss>

A2S/q_network_lossжЇ>о%k{+       УK	щзПzжA*

A2S/average_reward_1.2AГР%+       УK	vfоПzжA*

A2S/average_reward_1Жж2AsKO+       УK	иDхПzжA*

A2S/average_reward_193AГuћ+       УK	же	ПzжA*

A2S/average_reward_1  ЃAѓ5s5+       УK	p
ПzжA*

A2S/average_reward_1ХNмAMсоu+       УK	%uПzжA*

A2S/average_reward_1јрB5и
k+       УK	ОbПzжA*

A2S/average_reward_1=P5B0Zђђ+       УK	фIПzжA*

A2S/average_reward_1xx3BтТАд+       УK	[ЎПzжA*

A2S/average_reward_1зLBЯЬ+       УK	 ЎПzжA*

A2S/average_reward_1ёiB:а+       УK	F,ИПzжA*

A2S/average_reward_1ъЩB+Т+       УK	ШППzжA*

A2S/average_reward_1r\B*C5+       УK	ZвУПzжA*

A2S/average_reward_1#B\О+       УK	ЮЫПzжA*

A2S/average_reward_1|B)т=Ї+       УK	АрбПzжA*

A2S/average_reward_1ЭЬB#CИ+       УK	аЄиПzжA*

A2S/average_reward_1l(B) 6+       УK	нрПzжA*

A2S/average_reward_1BO>T+       УK	ж)ПzжA*

A2S/average_reward_1oЄB>{2+       УK	Њћ.ПzжA*

A2S/average_reward_1§ЎЂBmОЗE+       УK	3 5ПzжA*

A2S/average_reward_1  ЁB`qЙ+       УK	а<ПzжA*

A2S/average_reward_1Ј[BgўЭ+       УK	ЛCПzжA*

A2S/average_reward_1KдBGlvЉ+       УK	]ДПzжA*

A2S/average_reward_1PТЋBќmO +       УK	9МПzжA*

A2S/average_reward_11ЊB
В+       УK	ЏdУПzжA*

A2S/average_reward_1``ЈB`Сn+       УK	.ЯПzжA*

A2S/average_reward_1єЇB)д+       УK	З}ПzжA*

A2S/average_reward_1ЖГBUњє+       УK	XЃПzжA*

A2S/average_reward_1ЃKБBЩTМ+       УK	­7 ПzжA*

A2S/average_reward_1єЃЏBХЧd+       УK	T;WПzжA*

A2S/average_reward_1СЛBВ"$|+       УK	У}ПzжA*

A2S/average_reward_1г/ХBFз+       УK	Ш3ПzжA*

A2S/average_reward_1^УBКЈ2M+       УK	xПzжA*

A2S/average_reward_1cЫBФ&q+       УK	чПzжA*

A2S/average_reward_16ъЩB_oГ+       УK	ПzжA*

A2S/average_reward_1ШBвД+       УK	ЋќПzжA*

A2S/average_reward_1 PЦB*>+       УK	эxаПzжA*

A2S/average_reward_1ЮђаBКn+       УK	6кПzжA*

A2S/average_reward_1Ь)ЯBi.S+       УK	Ѓ
<ПzжA*

A2S/average_reward_1лB%фЃБ+       УK	HЃYПzжA*

A2S/average_reward_1зЃфBЉі<}+       УK	D`ПzжA*

A2S/average_reward_1qНфB	QЧГ+       УK	CзfПzжA*

A2S/average_reward_1)мфBhЩ+       УK	HmПzжA*

A2S/average_reward_1сњфBXDYG+       УK	тПzжA*

A2S/average_reward_1ЭL№B>0+       УK	РGПzжA*

A2S/average_reward_1
W№Bl|Бo+       УK	ќПzжA*

A2S/average_reward_1k№BашЅ+       УK	KАПzжA*

A2S/average_reward_1ыњBIїЕо+       УK	6ЙПzжA*

A2S/average_reward_1{ћBчк+       УK	pСПzжA*

A2S/average_reward_1ћBзгQ+       УK	HoЫПzжA*

A2S/average_reward_1q=ћB'>Fr+       УK	јегПzжA*

A2S/average_reward_1ЭLћBЖПљю+       УK	mK8!ПzжA*

A2S/average_reward_1ЭLCђ~нЧ+       УK	Dо?!ПzжA*

A2S/average_reward_1ьQCо9;+       УK	ФG!ПzжA*

A2S/average_reward_1HaCЌб+       УK	~СN!ПzжA*

A2S/average_reward_1ffC
тЌU+       УK	U!ПzжA*

A2S/average_reward_1kCм++       УK	0nN"ПzжA*

A2S/average_reward_1Єp	CБЉыє+       УK	ЄЇV"ПzжA*

A2S/average_reward_1Rx	CЫ%+       УK	nb_"ПzжA*

A2S/average_reward_1q}	C[И{E+       УK	}чg"ПzжA*

A2S/average_reward_1Э	CГќЩ       L	vM	ђ#ПzжAm*

A2S/klнГK>

A2S/average_advantageЉь?

A2S/policy_network_loss[7?

A2S/value_network_lossЏаB

A2S/q_network_lossyв BЋyIx+       УK	wі#ПzжAm*

A2S/average_reward_1Э	CaЬ+       УK	oгј#ПzжAm*

A2S/average_reward_1Ў	C{сP+       УK	sїћ#ПzжAm*

A2S/average_reward_1q}	Cѓї +       УK	^ѓў#ПzжAm*

A2S/average_reward_1сz	CюQМ+       УK	А)$ПzжAm*

A2S/average_reward_1Ў	CФ$+       УK	Х$ПzжAm*

A2S/average_reward_1\	Ch\+       УK	$ПzжAm*

A2S/average_reward_1	CIѕЦ+       УK	У&$ПzжAm*

A2S/average_reward_1Ў	Chй+       УK	оl$ПzжAm*

A2S/average_reward_1	C@у2Г+       УK	ч$ПzжAm*

A2S/average_reward_1\	CЁИ§+       УK	GD$ПzжAm*

A2S/average_reward_1=	CsГeZ+       УK	Ќг#$ПzжAm*

A2S/average_reward_1=	C!+       УK	)$ПzжAm*

A2S/average_reward_1=	C№GQ*+       УK	ДЌ,$ПzжAm*

A2S/average_reward_1 	C5л~S+       УK	н1$ПzжAm*

A2S/average_reward_1=	Cц8`Ж+       УK	"6$ПzжAm*

A2S/average_reward_1Ў	CПхЇ!+       УK	mp8$ПzжAm*

A2S/average_reward_1Уu	Cюї§+       УK	zў<$ПzжAm*

A2S/average_reward_1Rx	CР+       УK	W7B$ПzжAm*

A2S/average_reward_1Rx	C)EЌT+       УK	G$ПzжAm*

A2S/average_reward_1Єp	C]+       УK	%L$ПzжAm*

A2S/average_reward_1k	CFJЩ<+       УK	Р§P$ПzжAm*

A2S/average_reward_1k	CgЗF+       УK	ЉT$ПzжAm*

A2S/average_reward_1И^	C _Фэ+       УK	'Z$ПzжAm*

A2S/average_reward_1)\	CnзФ+       УK	ц]$ПzжAm*

A2S/average_reward_1Y	CF{Љ+       УK	E6d$ПzжAm*

A2S/average_reward_1
W	CCДОЂ+       УK	Яj$ПzжAm*

A2S/average_reward_1Y	CХ=+       УK	Мn$ПzжAm*

A2S/average_reward_1
W	C=V,ш+       УK	Qw$ПzжAm*

A2S/average_reward_1Y	CeqW+       УK	X	}$ПzжAm*

A2S/average_reward_1ьQ	CЈЙ+       УK	W$ПzжAm*

A2S/average_reward_1
W	CЂWП6+       УK	;$ПzжAm*

A2S/average_reward_1ьQ	C)Џ6l+       УK	&$ПzжAm*

A2S/average_reward_1=J	CеІ+       УK	Л$ПzжAm*

A2S/average_reward_1ЎG	CEЂЧя+       УK	Fй$ПzжAm*

A2S/average_reward_1 @	CHэ'+       УK	Iл$ПzжAm*

A2S/average_reward_1=J	CрL"+       УK	вtЅ$ПzжAm*

A2S/average_reward_1ЭL	CtГиН+       УK	ЄЉ$ПzжAm*

A2S/average_reward_1E	CџZЏ+       УK	pтЎ$ПzжAm*

A2S/average_reward_1E	Cю9-)+       УK	5Д$ПzжAm*

A2S/average_reward_1с:	C-§Ђ+       УK	Ы>Й$ПzжAm*

A2S/average_reward_1Э	CЋОl+       УK	aЅП$ПzжAm*

A2S/average_reward_1	CIc	й+       УK	|џФ$ПzжAm*

A2S/average_reward_1УѕC0rHX+       УK	 LЩ$ПzжAm*

A2S/average_reward_1\CБ"х+       УK	ѓЉа$ПzжAm*

A2S/average_reward_1ЎќB[Уc+       УK	чи$ПzжAm*

A2S/average_reward_1ЎюBЅн+       УK	Ўс$ПzжAm*

A2S/average_reward_1
WфBW~T+       УK	тGш$ПzжAm*

A2S/average_reward_1ЎGфBѕ@ђ+       УK	AЭэ$ПzжAm*

A2S/average_reward_1лBЗH2+       УK	Ђє$ПzжAm*

A2S/average_reward_1і(аBYаy6+       УK	 6ј$ПzжAm*

A2S/average_reward_1=
ХBwъ]+       УK	`Йў$ПzжAm*

A2S/average_reward_1  ХBЈn2m+       УK	юЮ%ПzжAm*

A2S/average_reward_1{КBТzЎ=+       УK	Ѕр
%ПzжAm*

A2S/average_reward_1R8АB$ц*^+       УK	ј%ПzжAm*

A2S/average_reward_1ИАBаОya+       УK	RЁ%ПzжAm*

A2S/average_reward_1  АBгцЁ+       УK	SЪ%ПzжAm*

A2S/average_reward_1ыЏBњW/+       УK	p%ПzжAm*

A2S/average_reward_1  ЃBлА_Ф+       УK	Nб%%ПzжAm*

A2S/average_reward_1сњЂBЎ|§з+       УK	&'-%ПzжAm*

A2S/average_reward_1УѕЂBпИ{       L	vM	CА&ПzжAќq*

A2S/kl,ВЮ<

A2S/average_advantageTЋН

A2S/policy_network_lossu#nО

A2S/value_network_loss	щR>

A2S/q_network_lossхс9>Вcљ+       УK	Ѕ}З&ПzжAќq*

A2S/average_reward_1=
ЃBnWo+       УK	OЯ&ПzжAќq*

A2S/average_reward_1ьбЃBKМј+       УK	Јїр&ПzжAќq*

A2S/average_reward_1УuBL Њ+       УK	ё&ПzжAќq*

A2S/average_reward_1ЭЬB`'Шь+       УK	Цг'ПzжAќq*

A2S/average_reward_1q=B{ +=+       УK	Ш'ПzжAќq*

A2S/average_reward_1ИBЙО+       УK	Рѕ'ПzжAќq*

A2S/average_reward_1ffB6ІЇm+       УK	Т'ПzжAќq*

A2S/average_reward_1kB5NУ+       УK	CwL'ПzжAќq*

A2S/average_reward_1=
BЎНњС+       УK	fb'ПzжAќq*

A2S/average_reward_1
зBЬб+       УK	fFi'ПzжAќq*

A2S/average_reward_1{pB>м8+       УK	QХz'ПzжAќq*

A2S/average_reward_1УѕpBР|+       УK	жX'ПzжAќq*

A2S/average_reward_1^BлЬ+       УK	Jр'ПzжAќq*

A2S/average_reward_1ff^BїО+       УK	WЛ'ПzжAќq*

A2S/average_reward_1і(aBКbд+       УK	вЛЦ'ПzжAќq*

A2S/average_reward_1)\aBQЏЧ+       УK	7т'ПzжAќq*

A2S/average_reward_1УѕJBGOє+       УK	&є'ПzжAќq*

A2S/average_reward_1RИKBl+       УK	гњћ'ПzжAќq*

A2S/average_reward_1Т0B'9й+       УK	R	(ПzжAќq*

A2S/average_reward_1ЄpB)+       УK	O)(ПzжAќq*

A2S/average_reward_1ьQBШDИT+       УK	(н>(ПzжAќq*

A2S/average_reward_133BQyлЯ+       УK	G(ПzжAќq*

A2S/average_reward_133B4їЉi+       УK	Ћi(ПzжAќq*

A2S/average_reward_1\BЮ/ь+       УK	нu(ПzжAќq*

A2S/average_reward_1ЭЬBОЏ+       УK	Аg(ПzжAќq*

A2S/average_reward_1И	B_(.Э+       УK	НнЖ(ПzжAќq*

A2S/average_reward_1HсюAa+       УK	щжб(ПzжAќq*

A2S/average_reward_1ТёAцq9O+       УK	ълр(ПzжAќq*

A2S/average_reward_133ѓA*єc+       УK	Y^ъ(ПzжAќq*

A2S/average_reward_133ѓAbт+       УK	Vыў(ПzжAќq*

A2S/average_reward_1)\ѕAЈAЙg+       УK	й)ПzжAќq*

A2S/average_reward_1зЃРAУзЋz+       УK	G`)ПzжAќq*

A2S/average_reward_1зЃРA>{ВБ+       УK	G)ПzжAќq*

A2S/average_reward_1)\СAR.)+       УK	Р[+)ПzжAќq*

A2S/average_reward_1ffТAљй+       УK	B)ПzжAќq*

A2S/average_reward_1зЃФA ф+       УK	ИU)ПzжAќq*

A2S/average_reward_1сzAнїи+       УK	ЋЪh)ПzжAќq*

A2S/average_reward_1{ A`Dњ+       УK	|)ПzжAќq*

A2S/average_reward_1ЄpЁANР +       УK	ЖЎ)ПzжAќq*

A2S/average_reward_1
зЁAсG+       УK	_є)ПzжAќq*

A2S/average_reward_1ьQЂAЙ+Вs+       УK	J'Ў)ПzжAќq*

A2S/average_reward_1\ІAyь+       УK	Ш`Х)ПzжAќq*

A2S/average_reward_1ЎЉA?Ј+       УK	Ы т)ПzжAќq*

A2S/average_reward_1Ў­Apsbв+       УK	FCъ)ПzжAќq*

A2S/average_reward_1{ЎAj\l+       УK	ЉМђ)ПzжAќq*

A2S/average_reward_1зЃЎAЄЬL+       УK	Г<ћ)ПzжAќq*

A2S/average_reward_1ЎGЏA
Щж+       УK	mб*ПzжAќq*

A2S/average_reward_1  АA	Ре+       УK	NK#*ПzжAќq*

A2S/average_reward_1ГAъO+       УK	ь8*ПzжAќq*

A2S/average_reward_1сzЖA~с+       УK	?*ПzжAќq*

A2S/average_reward_1УѕЖA@Нў+       УK	ЅE*ПzжAќq*

A2S/average_reward_133ЗATw+       УK	KeZ*ПzжAќq*

A2S/average_reward_1ЎЙA?э>7+       УK	ОЈr*ПzжAќq*

A2S/average_reward_1\МAШdж+       УK	z3}*ПzжAќq*

A2S/average_reward_1ИНAњgfЏ+       УK	Ј*ПzжAќq*

A2S/average_reward_1  ОA!АЗЉ+       УK	KTЃ*ПzжAќq*

A2S/average_reward_1ЎСAќP.+       УK	щњГ*ПzжAќq*

A2S/average_reward_1HсТA6iЪ+       УK	ЖcЛ*ПzжAќq*

A2S/average_reward_1ЭЬТAJєЕp+       УK	%9Ш*ПzжAќq*

A2S/average_reward_1ыУAW+       УK	dз*ПzжAќq*

A2S/average_reward_1ЎХA]к+       УK	їGт*ПzжAќq*

A2S/average_reward_1сzЦAОQј+       УK	ї*ПzжAќq*

A2S/average_reward_133ЩAZ§т	+       УK	EJ*+ПzжAќq*

A2S/average_reward_1ьQаAxёZk+       УK	дё1+ПzжAќq*

A2S/average_reward_1УѕаA5,й+       УK	Ў<+ПzжAќq*

A2S/average_reward_1ыбAѓ[+       УK	G+ПzжAќq*

A2S/average_reward_1ЭЬвAцЬ+       УK	РяT+ПzжAќq*

A2S/average_reward_1\дAlТ+       УK	Jo+ПzжAќq*

A2S/average_reward_1HсжA?ч+       УK	x+ПzжAќq*

A2S/average_reward_1УѕжAr`ю+       УK	f+ПzжAќq*

A2S/average_reward_1=
йAwњ:+       УK	ЉHЂ+ПzжAќq*

A2S/average_reward_1ЄpлAђѓ+       УK	hЈ+ПzжAќq*

A2S/average_reward_1ЎлAsє+       УK	yVТ+ПzжAќq*

A2S/average_reward_1ЄpпAјПь+       УK	й1г+ПzжAќq*

A2S/average_reward_1ИсAr+       УK	YКл+ПzжAќq*

A2S/average_reward_1ЄpсAњZф+       УK	Ђrљ+ПzжAќq*

A2S/average_reward_1ЄpхAOr:+       УK	ЮEў+ПzжAќq*

A2S/average_reward_1ЎхAЖћ/Ъ+       УK	ж	,ПzжAќq*

A2S/average_reward_1ЄpчAКi+       УK	Nа,ПzжAќq*

A2S/average_reward_1і(шAЇ@!        фv@ѓ	ёц-ПzжAў*

A2S/kl6­=

A2S/average_advantageњњ>

A2S/policy_network_lossZО

A2S/value_network_lossAA

A2S/q_network_loss5*ЊA ёљI,       єЎЬE	л-ПzжAў*

A2S/average_reward_1ЭЬшAvfОC,       єЎЬE	\к-ПzжAў*

A2S/average_reward_1\ђAЕЇm,       єЎЬE	bQ.ПzжAў*

A2S/average_reward_1)\їAщ>м,       єЎЬE	;З.ПzжAў*

A2S/average_reward_1
зїA нpM,       єЎЬE	|S.ПzжAў*

A2S/average_reward_1і(B\cu,       єЎЬE	тJz.ПzжAў*

A2S/average_reward_1
зBUгГя,       єЎЬE	КЁЎ.ПzжAў*

A2S/average_reward_1=
BZt,       єЎЬE	/ПzжAў*

A2S/average_reward_1УѕB!ЋЊ,       єЎЬE	,3/ПzжAў*

A2S/average_reward_1ЄpB­Џ%,       єЎЬE	<:/ПzжAў*

A2S/average_reward_1\BеM",       єЎЬE	њA/ПzжAў*

A2S/average_reward_1УѕBaк­,       єЎЬE	pEI/ПzжAў*

A2S/average_reward_1і(BЛ,       єЎЬE	/ПzжAў*

A2S/average_reward_1ЄpBJ­,       єЎЬE	%Ћг/ПzжAў*

A2S/average_reward_1ЭЬBbВм,       єЎЬE	У0ПzжAў*

A2S/average_reward_1ЎG!Bm2щ,       єЎЬE	ЎU0ПzжAў*

A2S/average_reward_1
з&BCЮЃ№,       єЎЬE	И\0ПzжAў*

A2S/average_reward_1Уѕ&B№",       єЎЬE	е[0ПzжAў*

A2S/average_reward_1Hс*BwЎt,       єЎЬE	џбк0ПzжAў*

A2S/average_reward_1RИ.B №,,       єЎЬE		 ч0ПzжAў*

A2S/average_reward_1ы.BЛY),       єЎЬE	Бъя0ПzжAў*

A2S/average_reward_1ЭЬ.BЦЯ,       єЎЬE	ІМњ0ПzжAў*

A2S/average_reward_1q=-BэIљщ,       єЎЬE	T=1ПzжAў*

A2S/average_reward_1{1Bљц,       єЎЬE	 1ПzжAў*

A2S/average_reward_1
з5BJZ,       єЎЬE	ЪY1ПzжAў*

A2S/average_reward_1q=5BА,       єЎЬE	­вэ1ПzжAў*

A2S/average_reward_1ьQ:BУТд,       єЎЬE	@2ПzжAў*

A2S/average_reward_1ьQ?Bv[',       єЎЬE	r,H2ПzжAў*

A2S/average_reward_1ff?B$,       єЎЬE	L2ПzжAў*

A2S/average_reward_1Hс;Bћ]мд,       єЎЬE	R2ПzжAў*

A2S/average_reward_1)\?B~=,       єЎЬE	iЌ2ПzжAў*

A2S/average_reward_1ff?BсRYэ,       єЎЬE	:г2ПzжAў*

A2S/average_reward_1ыCBOT,       єЎЬE	Ык2ПzжAў*

A2S/average_reward_1ЭЬCB:&1,       єЎЬE	]ѓс2ПzжAў*

A2S/average_reward_1CB&,       єЎЬE	Гх3ПzжAў*

A2S/average_reward_1і(EB("Ч,       єЎЬE	iМh3ПzжAў*

A2S/average_reward_1і(JBЊ/,       єЎЬE	 бs3ПzжAў*

A2S/average_reward_1RИHBГвs,       єЎЬE	1 Ю3ПzжAў*

A2S/average_reward_1ЭЬMBmП"f,       єЎЬE	;ъ3ПzжAў*

A2S/average_reward_133OB=M^,       єЎЬE	їm>4ПzжAў*

A2S/average_reward_1ьQTBв9,       єЎЬE	еh4ПzжAў*

A2S/average_reward_1ffWBйКуЦ,       єЎЬE	Кг4ПzжAў*

A2S/average_reward_1ЭЬ[BXћВV,       єЎЬE	u  5ПzжAў*

A2S/average_reward_1ЄpaBJЦ5;,       єЎЬE	Сg5ПzжAў*

A2S/average_reward_1ьQdBЏЌ%Д,       єЎЬE	№Љ5ПzжAў*

A2S/average_reward_1  iBгЏ1,       єЎЬE	Г5ПzжAў*

A2S/average_reward_1ТhBsЫЯЁ,       єЎЬE	ХUё5ПzжAў*

A2S/average_reward_1УѕiB	§Z,       єЎЬE	Hіј5ПzжAў*

A2S/average_reward_1\hBAсЗ,       єЎЬE	{Ї6ПzжAў*

A2S/average_reward_1  hBЈЅT,       єЎЬE	мO6ПzжAў*

A2S/average_reward_1сzmBlцn,       єЎЬE	X6ПzжAў*

A2S/average_reward_1ЄplBГйLѕ,       єЎЬE	1п 6ПzжAў*

A2S/average_reward_1зЃqB§ЙTН,       єЎЬE	Ј6ПzжAў*

A2S/average_reward_1RИqBЕ.К,       єЎЬE	5В6ПzжAў*

A2S/average_reward_1qBЋZо,       єЎЬE	щЌѕ6ПzжAў*

A2S/average_reward_1ЎGvBЃЏLy,       єЎЬE	њЭ§6ПzжAў*

A2S/average_reward_133uBфЇ	H,       єЎЬE	+	B7ПzжAў*

A2S/average_reward_1RИxBљ>о,       єЎЬE	Х7ПzжAў*

A2S/average_reward_1ЭЬ{B
`R,       єЎЬE	ив7ПzжAў*

A2S/average_reward_1і(Bдqв№,       єЎЬE	yQ8ПzжAў*

A2S/average_reward_1HaBR{pR,       єЎЬE	іY8ПzжAў*

A2S/average_reward_1HсB2xЬQ,       єЎЬE	Сыb8ПzжAў*

A2S/average_reward_1УѕBш.,       єЎЬE	ecl8ПzжAў*

A2S/average_reward_1
WBJФЃ,       єЎЬE	ot8ПzжAў*

A2S/average_reward_1сzBGђ3в,       єЎЬE	Y|8ПzжAў*

A2S/average_reward_1 Bkа* ,       єЎЬE	Ю8ПzжAў*

A2S/average_reward_1ffBDўв,       єЎЬE	яџа8ПzжAў*

A2S/average_reward_1ЎGBЉRв,       єЎЬE	UЗк8ПzжAў*

A2S/average_reward_1з#BD;ь,       єЎЬE	Флх8ПzжAў*

A2S/average_reward_1
WBDGъ,       єЎЬE	ѕ{ю8ПzжAў*

A2S/average_reward_1qНB1§,       єЎЬE	ј8ПzжAў*

A2S/average_reward_1ТB:Щwћ,       єЎЬE	Yъ]9ПzжAў*

A2S/average_reward_1)мB{№ 	,       єЎЬE	ЩЋ9ПzжAў*

A2S/average_reward_1)мBёў,       єЎЬE	эpЖ9ПzжAў*

A2S/average_reward_1
WB№ЮА,       єЎЬE	д:ПzжAў*

A2S/average_reward_1ТB(ь,       єЎЬE	G#:ПzжAў*

A2S/average_reward_1\BЎЕЬі,       єЎЬE	ъ,:ПzжAў*

A2S/average_reward_1ЭЬBЇ*aі,       єЎЬE	Rх:ПzжAў*

A2S/average_reward_1сzB;zу,       єЎЬE	{и:ПzжAў*

A2S/average_reward_1{BтFDФ,       єЎЬE	щ$;ПzжAў*

A2S/average_reward_1{BѕБ#,       єЎЬE	yN_;ПzжAў*

A2S/average_reward_1)\BаЈX,       єЎЬE	 2;ПzжAў*

A2S/average_reward_1 Bќ@b,       єЎЬE	%ЫЉ;ПzжAў*

A2S/average_reward_1  B  ю?,       єЎЬE	ФкЕ;ПzжAў*

A2S/average_reward_1)\BЗI`џ,       єЎЬE	ХП;ПzжAў*

A2S/average_reward_1HaBGНўО,       єЎЬE	ќCЪ;ПzжAў*

A2S/average_reward_1q=B~Ќ(