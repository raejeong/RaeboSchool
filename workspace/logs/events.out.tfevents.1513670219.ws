       гK"	  └Т1О╓Abrain.Event:2	iця     уBл&	mс╚Т1О╓A"тн0
R
learning_ratePlaceholder*
_output_shapes
:*
shape:*
dtype0
o
observationsPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
v
target_observationsPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
j
actionsPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
q
target_actionsPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
r
target_q_valuesPlaceholder*'
_output_shapes
:         *
shape:         *
dtype0
j
rewardsPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
}
concatConcatV2observationsactionsconcat/axis*'
_output_shapes
:         *

Tidx0*
T0*
N
O
concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
П
concat_1ConcatV2target_observationstarget_actionsconcat_1/axis*
T0*
N*'
_output_shapes
:         *

Tidx0
┘
Jcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
valueB"   А   *
dtype0
╦
Hcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/minConst*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╦
Hcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxConst*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╜
Rcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformJcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
seed2*
dtype0*
_output_shapes
:	А*

seed
┬
Hcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/subSubHcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxHcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
_output_shapes
: 
╒
Hcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulMulRcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformHcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w
╟
Dcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniformAddHcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulHcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
_output_shapes
:	А
▌
)current_q_network/current_q_network/fc0/w
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
	container *
shape:	А
╝
0current_q_network/current_q_network/fc0/w/AssignAssign)current_q_network/current_q_network/fc0/wDcurrent_q_network/current_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes
:	А
═
.current_q_network/current_q_network/fc0/w/readIdentity)current_q_network/current_q_network/fc0/w*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
_output_shapes
:	А
╚
;current_q_network/current_q_network/fc0/b/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
valueBА*    *
dtype0*
_output_shapes	
:А
╒
)current_q_network/current_q_network/fc0/b
VariableV2*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
	container *
shape:А*
dtype0
п
0current_q_network/current_q_network/fc0/b/AssignAssign)current_q_network/current_q_network/fc0/b;current_q_network/current_q_network/fc0/b/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
validate_shape(
╔
.current_q_network/current_q_network/fc0/b/readIdentity)current_q_network/current_q_network/fc0/b*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
_output_shapes	
:А
│
current_q_network/MatMulMatMulconcat.current_q_network/current_q_network/fc0/w/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
Щ
current_q_network/addAddcurrent_q_network/MatMul.current_q_network/current_q_network/fc0/b/read*(
_output_shapes
:         А*
T0
╢
2current_q_network/LayerNorm/beta/Initializer/zerosConst*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
├
 current_q_network/LayerNorm/beta
VariableV2*
shared_name *3
_class)
'%loc:@current_q_network/LayerNorm/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Л
'current_q_network/LayerNorm/beta/AssignAssign current_q_network/LayerNorm/beta2current_q_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta
о
%current_q_network/LayerNorm/beta/readIdentity current_q_network/LayerNorm/beta*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
_output_shapes	
:А
╖
2current_q_network/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:А*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
valueBА*  А?
┼
!current_q_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
	container *
shape:А
О
(current_q_network/LayerNorm/gamma/AssignAssign!current_q_network/LayerNorm/gamma2current_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
▒
&current_q_network/LayerNorm/gamma/readIdentity!current_q_network/LayerNorm/gamma*
_output_shapes	
:А*
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma
Д
:current_q_network/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
╥
(current_q_network/LayerNorm/moments/meanMeancurrent_q_network/add:current_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
Ь
0current_q_network/LayerNorm/moments/StopGradientStopGradient(current_q_network/LayerNorm/moments/mean*'
_output_shapes
:         *
T0
╞
5current_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencecurrent_q_network/add0current_q_network/LayerNorm/moments/StopGradient*
T0*(
_output_shapes
:         А
И
>current_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
·
,current_q_network/LayerNorm/moments/varianceMean5current_q_network/LayerNorm/moments/SquaredDifference>current_q_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
p
+current_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╜
)current_q_network/LayerNorm/batchnorm/addAdd,current_q_network/LayerNorm/moments/variance+current_q_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
С
+current_q_network/LayerNorm/batchnorm/RsqrtRsqrt)current_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
╕
)current_q_network/LayerNorm/batchnorm/mulMul+current_q_network/LayerNorm/batchnorm/Rsqrt&current_q_network/LayerNorm/gamma/read*
T0*(
_output_shapes
:         А
з
+current_q_network/LayerNorm/batchnorm/mul_1Mulcurrent_q_network/add)current_q_network/LayerNorm/batchnorm/mul*
T0*(
_output_shapes
:         А
║
+current_q_network/LayerNorm/batchnorm/mul_2Mul(current_q_network/LayerNorm/moments/mean)current_q_network/LayerNorm/batchnorm/mul*
T0*(
_output_shapes
:         А
╖
)current_q_network/LayerNorm/batchnorm/subSub%current_q_network/LayerNorm/beta/read+current_q_network/LayerNorm/batchnorm/mul_2*
T0*(
_output_shapes
:         А
╜
+current_q_network/LayerNorm/batchnorm/add_1Add+current_q_network/LayerNorm/batchnorm/mul_1)current_q_network/LayerNorm/batchnorm/sub*(
_output_shapes
:         А*
T0
~
current_q_network/TanhTanh+current_q_network/LayerNorm/batchnorm/add_1*(
_output_shapes
:         А*
T0
┘
Jcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
valueB"А   @   *
dtype0*
_output_shapes
:
╦
Hcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/minConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╦
Hcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╜
Rcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformJcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А@*

seed*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
seed25
┬
Hcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/subSubHcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxHcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
_output_shapes
: *
T0
╒
Hcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulMulRcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformHcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
_output_shapes
:	А@
╟
Dcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniformAddHcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulHcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
:	А@*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w
▌
)current_q_network/current_q_network/fc1/w
VariableV2*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
	container *
shape:	А@*
dtype0*
_output_shapes
:	А@
╝
0current_q_network/current_q_network/fc1/w/AssignAssign)current_q_network/current_q_network/fc1/wDcurrent_q_network/current_q_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@
═
.current_q_network/current_q_network/fc1/w/readIdentity)current_q_network/current_q_network/fc1/w*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
_output_shapes
:	А@
╞
;current_q_network/current_q_network/fc1/b/Initializer/zerosConst*
_output_shapes
:@*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0
╙
)current_q_network/current_q_network/fc1/b
VariableV2*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
о
0current_q_network/current_q_network/fc1/b/AssignAssign)current_q_network/current_q_network/fc1/b;current_q_network/current_q_network/fc1/b/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
╚
.current_q_network/current_q_network/fc1/b/readIdentity)current_q_network/current_q_network/fc1/b*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
_output_shapes
:@
─
current_q_network/MatMul_1MatMulcurrent_q_network/Tanh.current_q_network/current_q_network/fc1/w/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b( *
T0
Ь
current_q_network/add_1Addcurrent_q_network/MatMul_1.current_q_network/current_q_network/fc1/b/read*
T0*'
_output_shapes
:         @
╕
4current_q_network/LayerNorm_1/beta/Initializer/zerosConst*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
┼
"current_q_network/LayerNorm_1/beta
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
	container 
Т
)current_q_network/LayerNorm_1/beta/AssignAssign"current_q_network/LayerNorm_1/beta4current_q_network/LayerNorm_1/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta
│
'current_q_network/LayerNorm_1/beta/readIdentity"current_q_network/LayerNorm_1/beta*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
_output_shapes
:@*
T0
╣
4current_q_network/LayerNorm_1/gamma/Initializer/onesConst*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╟
#current_q_network/LayerNorm_1/gamma
VariableV2*
shared_name *6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
Х
*current_q_network/LayerNorm_1/gamma/AssignAssign#current_q_network/LayerNorm_1/gamma4current_q_network/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╢
(current_q_network/LayerNorm_1/gamma/readIdentity#current_q_network/LayerNorm_1/gamma*
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
_output_shapes
:@
Ж
<current_q_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╪
*current_q_network/LayerNorm_1/moments/meanMeancurrent_q_network/add_1<current_q_network/LayerNorm_1/moments/mean/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
а
2current_q_network/LayerNorm_1/moments/StopGradientStopGradient*current_q_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
╦
7current_q_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencecurrent_q_network/add_12current_q_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
К
@current_q_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
А
.current_q_network/LayerNorm_1/moments/varianceMean7current_q_network/LayerNorm_1/moments/SquaredDifference@current_q_network/LayerNorm_1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
r
-current_q_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
├
+current_q_network/LayerNorm_1/batchnorm/addAdd.current_q_network/LayerNorm_1/moments/variance-current_q_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
Х
-current_q_network/LayerNorm_1/batchnorm/RsqrtRsqrt+current_q_network/LayerNorm_1/batchnorm/add*'
_output_shapes
:         *
T0
╜
+current_q_network/LayerNorm_1/batchnorm/mulMul-current_q_network/LayerNorm_1/batchnorm/Rsqrt(current_q_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
м
-current_q_network/LayerNorm_1/batchnorm/mul_1Mulcurrent_q_network/add_1+current_q_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
┐
-current_q_network/LayerNorm_1/batchnorm/mul_2Mul*current_q_network/LayerNorm_1/moments/mean+current_q_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
╝
+current_q_network/LayerNorm_1/batchnorm/subSub'current_q_network/LayerNorm_1/beta/read-current_q_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
┬
-current_q_network/LayerNorm_1/batchnorm/add_1Add-current_q_network/LayerNorm_1/batchnorm/mul_1+current_q_network/LayerNorm_1/batchnorm/sub*
T0*'
_output_shapes
:         @
Б
current_q_network/Tanh_1Tanh-current_q_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
┘
Jcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
╦
Hcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/minConst*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
valueB
 *═╠╠╜*
dtype0*
_output_shapes
: 
╦
Hcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/maxConst*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
╝
Rcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformJcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
seed2\
┬
Hcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/subSubHcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/maxHcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
_output_shapes
: 
╘
Hcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/mulMulRcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformHcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
_output_shapes

:@
╞
Dcurrent_q_network/current_q_network/out/w/Initializer/random_uniformAddHcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/mulHcurrent_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
_output_shapes

:@
█
)current_q_network/current_q_network/out/w
VariableV2*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
╗
0current_q_network/current_q_network/out/w/AssignAssign)current_q_network/current_q_network/out/wDcurrent_q_network/current_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
╠
.current_q_network/current_q_network/out/w/readIdentity)current_q_network/current_q_network/out/w*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
_output_shapes

:@
╞
;current_q_network/current_q_network/out/b/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
╙
)current_q_network/current_q_network/out/b
VariableV2*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
о
0current_q_network/current_q_network/out/b/AssignAssign)current_q_network/current_q_network/out/b;current_q_network/current_q_network/out/b/Initializer/zeros*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
╚
.current_q_network/current_q_network/out/b/readIdentity)current_q_network/current_q_network/out/b*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
_output_shapes
:
╞
current_q_network/MatMul_2MatMulcurrent_q_network/Tanh_1.current_q_network/current_q_network/out/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Ь
current_q_network/add_2Addcurrent_q_network/MatMul_2.current_q_network/current_q_network/out/b/read*
T0*'
_output_shapes
:         
╒
Htarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
valueB"   А   *
dtype0
╟
Ftarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╟
Ftarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╖
Ptarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformHtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/shape*
_output_shapes
:	А*

seed*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
seed2l*
dtype0
║
Ftarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/subSubFtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/maxFtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
_output_shapes
: 
═
Ftarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/mulMulPtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/RandomUniformFtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
_output_shapes
:	А
┐
Btarget_q_network/target_q_network/fc0/w/Initializer/random_uniformAddFtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/mulFtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
_output_shapes
:	А
┘
'target_q_network/target_q_network/fc0/w
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
	container *
shape:	А
┤
.target_q_network/target_q_network/fc0/w/AssignAssign'target_q_network/target_q_network/fc0/wBtarget_q_network/target_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
validate_shape(*
_output_shapes
:	А
╟
,target_q_network/target_q_network/fc0/w/readIdentity'target_q_network/target_q_network/fc0/w*
_output_shapes
:	А*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w
─
9target_q_network/target_q_network/fc0/b/Initializer/zerosConst*:
_class0
.,loc:@target_q_network/target_q_network/fc0/b*
valueBА*    *
dtype0*
_output_shapes	
:А
╤
'target_q_network/target_q_network/fc0/b
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *:
_class0
.,loc:@target_q_network/target_q_network/fc0/b
з
.target_q_network/target_q_network/fc0/b/AssignAssign'target_q_network/target_q_network/fc0/b9target_q_network/target_q_network/fc0/b/Initializer/zeros*:
_class0
.,loc:@target_q_network/target_q_network/fc0/b*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
├
,target_q_network/target_q_network/fc0/b/readIdentity'target_q_network/target_q_network/fc0/b*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/b*
_output_shapes	
:А
▓
target_q_network/MatMulMatMulconcat_1,target_q_network/target_q_network/fc0/w/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
Х
target_q_network/addAddtarget_q_network/MatMul,target_q_network/target_q_network/fc0/b/read*(
_output_shapes
:         А*
T0
┤
1target_q_network/LayerNorm/beta/Initializer/zerosConst*2
_class(
&$loc:@target_q_network/LayerNorm/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
┴
target_q_network/LayerNorm/beta
VariableV2*
shared_name *2
_class(
&$loc:@target_q_network/LayerNorm/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
З
&target_q_network/LayerNorm/beta/AssignAssigntarget_q_network/LayerNorm/beta1target_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@target_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes	
:А
л
$target_q_network/LayerNorm/beta/readIdentitytarget_q_network/LayerNorm/beta*
_output_shapes	
:А*
T0*2
_class(
&$loc:@target_q_network/LayerNorm/beta
╡
1target_q_network/LayerNorm/gamma/Initializer/onesConst*3
_class)
'%loc:@target_q_network/LayerNorm/gamma*
valueBА*  А?*
dtype0*
_output_shapes	
:А
├
 target_q_network/LayerNorm/gamma
VariableV2*
shared_name *3
_class)
'%loc:@target_q_network/LayerNorm/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А
К
'target_q_network/LayerNorm/gamma/AssignAssign target_q_network/LayerNorm/gamma1target_q_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*3
_class)
'%loc:@target_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
о
%target_q_network/LayerNorm/gamma/readIdentity target_q_network/LayerNorm/gamma*
_output_shapes	
:А*
T0*3
_class)
'%loc:@target_q_network/LayerNorm/gamma
Г
9target_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╧
'target_q_network/LayerNorm/moments/meanMeantarget_q_network/add9target_q_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
Ъ
/target_q_network/LayerNorm/moments/StopGradientStopGradient'target_q_network/LayerNorm/moments/mean*'
_output_shapes
:         *
T0
├
4target_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencetarget_q_network/add/target_q_network/LayerNorm/moments/StopGradient*(
_output_shapes
:         А*
T0
З
=target_q_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ў
+target_q_network/LayerNorm/moments/varianceMean4target_q_network/LayerNorm/moments/SquaredDifference=target_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
o
*target_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
║
(target_q_network/LayerNorm/batchnorm/addAdd+target_q_network/LayerNorm/moments/variance*target_q_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:         *
T0
П
*target_q_network/LayerNorm/batchnorm/RsqrtRsqrt(target_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
╡
(target_q_network/LayerNorm/batchnorm/mulMul*target_q_network/LayerNorm/batchnorm/Rsqrt%target_q_network/LayerNorm/gamma/read*
T0*(
_output_shapes
:         А
д
*target_q_network/LayerNorm/batchnorm/mul_1Multarget_q_network/add(target_q_network/LayerNorm/batchnorm/mul*(
_output_shapes
:         А*
T0
╖
*target_q_network/LayerNorm/batchnorm/mul_2Mul'target_q_network/LayerNorm/moments/mean(target_q_network/LayerNorm/batchnorm/mul*
T0*(
_output_shapes
:         А
┤
(target_q_network/LayerNorm/batchnorm/subSub$target_q_network/LayerNorm/beta/read*target_q_network/LayerNorm/batchnorm/mul_2*(
_output_shapes
:         А*
T0
║
*target_q_network/LayerNorm/batchnorm/add_1Add*target_q_network/LayerNorm/batchnorm/mul_1(target_q_network/LayerNorm/batchnorm/sub*(
_output_shapes
:         А*
T0
|
target_q_network/TanhTanh*target_q_network/LayerNorm/batchnorm/add_1*(
_output_shapes
:         А*
T0
╒
Htarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
valueB"А   @   *
dtype0*
_output_shapes
:
╟
Ftarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╟
Ftarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/maxConst*
_output_shapes
: *:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
valueB
 *  А?*
dtype0
╕
Ptarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformHtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А@*

seed*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
seed2У
║
Ftarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/subSubFtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/maxFtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w
═
Ftarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/mulMulPtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/RandomUniformFtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
_output_shapes
:	А@
┐
Btarget_q_network/target_q_network/fc1/w/Initializer/random_uniformAddFtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/mulFtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
_output_shapes
:	А@
┘
'target_q_network/target_q_network/fc1/w
VariableV2*
shared_name *:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
	container *
shape:	А@*
dtype0*
_output_shapes
:	А@
┤
.target_q_network/target_q_network/fc1/w/AssignAssign'target_q_network/target_q_network/fc1/wBtarget_q_network/target_q_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@
╟
,target_q_network/target_q_network/fc1/w/readIdentity'target_q_network/target_q_network/fc1/w*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
_output_shapes
:	А@
┬
9target_q_network/target_q_network/fc1/b/Initializer/zerosConst*:
_class0
.,loc:@target_q_network/target_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
╧
'target_q_network/target_q_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@target_q_network/target_q_network/fc1/b*
	container *
shape:@
ж
.target_q_network/target_q_network/fc1/b/AssignAssign'target_q_network/target_q_network/fc1/b9target_q_network/target_q_network/fc1/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
┬
,target_q_network/target_q_network/fc1/b/readIdentity'target_q_network/target_q_network/fc1/b*
_output_shapes
:@*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/b
└
target_q_network/MatMul_1MatMultarget_q_network/Tanh,target_q_network/target_q_network/fc1/w/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b( *
T0
Ш
target_q_network/add_1Addtarget_q_network/MatMul_1,target_q_network/target_q_network/fc1/b/read*
T0*'
_output_shapes
:         @
╢
3target_q_network/LayerNorm_1/beta/Initializer/zerosConst*4
_class*
(&loc:@target_q_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
├
!target_q_network/LayerNorm_1/beta
VariableV2*
shared_name *4
_class*
(&loc:@target_q_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
О
(target_q_network/LayerNorm_1/beta/AssignAssign!target_q_network/LayerNorm_1/beta3target_q_network/LayerNorm_1/beta/Initializer/zeros*
T0*4
_class*
(&loc:@target_q_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
░
&target_q_network/LayerNorm_1/beta/readIdentity!target_q_network/LayerNorm_1/beta*
T0*4
_class*
(&loc:@target_q_network/LayerNorm_1/beta*
_output_shapes
:@
╖
3target_q_network/LayerNorm_1/gamma/Initializer/onesConst*5
_class+
)'loc:@target_q_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
┼
"target_q_network/LayerNorm_1/gamma
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *5
_class+
)'loc:@target_q_network/LayerNorm_1/gamma
С
)target_q_network/LayerNorm_1/gamma/AssignAssign"target_q_network/LayerNorm_1/gamma3target_q_network/LayerNorm_1/gamma/Initializer/ones*5
_class+
)'loc:@target_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
│
'target_q_network/LayerNorm_1/gamma/readIdentity"target_q_network/LayerNorm_1/gamma*
T0*5
_class+
)'loc:@target_q_network/LayerNorm_1/gamma*
_output_shapes
:@
Е
;target_q_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╒
)target_q_network/LayerNorm_1/moments/meanMeantarget_q_network/add_1;target_q_network/LayerNorm_1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
Ю
1target_q_network/LayerNorm_1/moments/StopGradientStopGradient)target_q_network/LayerNorm_1/moments/mean*'
_output_shapes
:         *
T0
╚
6target_q_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencetarget_q_network/add_11target_q_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
Й
?target_q_network/LayerNorm_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
¤
-target_q_network/LayerNorm_1/moments/varianceMean6target_q_network/LayerNorm_1/moments/SquaredDifference?target_q_network/LayerNorm_1/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
q
,target_q_network/LayerNorm_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *╠╝М+*
dtype0
└
*target_q_network/LayerNorm_1/batchnorm/addAdd-target_q_network/LayerNorm_1/moments/variance,target_q_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
У
,target_q_network/LayerNorm_1/batchnorm/RsqrtRsqrt*target_q_network/LayerNorm_1/batchnorm/add*'
_output_shapes
:         *
T0
║
*target_q_network/LayerNorm_1/batchnorm/mulMul,target_q_network/LayerNorm_1/batchnorm/Rsqrt'target_q_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
й
,target_q_network/LayerNorm_1/batchnorm/mul_1Multarget_q_network/add_1*target_q_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
╝
,target_q_network/LayerNorm_1/batchnorm/mul_2Mul)target_q_network/LayerNorm_1/moments/mean*target_q_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
╣
*target_q_network/LayerNorm_1/batchnorm/subSub&target_q_network/LayerNorm_1/beta/read,target_q_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
┐
,target_q_network/LayerNorm_1/batchnorm/add_1Add,target_q_network/LayerNorm_1/batchnorm/mul_1*target_q_network/LayerNorm_1/batchnorm/sub*'
_output_shapes
:         @*
T0

target_q_network/Tanh_1Tanh,target_q_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
╒
Htarget_q_network/target_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
╟
Ftarget_q_network/target_q_network/out/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
valueB
 *═╠╠╜*
dtype0*
_output_shapes
: 
╟
Ftarget_q_network/target_q_network/out/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
╖
Ptarget_q_network/target_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHtarget_q_network/target_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
seed2║
║
Ftarget_q_network/target_q_network/out/w/Initializer/random_uniform/subSubFtarget_q_network/target_q_network/out/w/Initializer/random_uniform/maxFtarget_q_network/target_q_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w
╠
Ftarget_q_network/target_q_network/out/w/Initializer/random_uniform/mulMulPtarget_q_network/target_q_network/out/w/Initializer/random_uniform/RandomUniformFtarget_q_network/target_q_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w
╛
Btarget_q_network/target_q_network/out/w/Initializer/random_uniformAddFtarget_q_network/target_q_network/out/w/Initializer/random_uniform/mulFtarget_q_network/target_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
_output_shapes

:@
╫
'target_q_network/target_q_network/out/w
VariableV2*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
│
.target_q_network/target_q_network/out/w/AssignAssign'target_q_network/target_q_network/out/wBtarget_q_network/target_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
validate_shape(*
_output_shapes

:@
╞
,target_q_network/target_q_network/out/w/readIdentity'target_q_network/target_q_network/out/w*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
_output_shapes

:@
┬
9target_q_network/target_q_network/out/b/Initializer/zerosConst*:
_class0
.,loc:@target_q_network/target_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
╧
'target_q_network/target_q_network/out/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@target_q_network/target_q_network/out/b*
	container 
ж
.target_q_network/target_q_network/out/b/AssignAssign'target_q_network/target_q_network/out/b9target_q_network/target_q_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/b*
validate_shape(
┬
,target_q_network/target_q_network/out/b/readIdentity'target_q_network/target_q_network/out/b*:
_class0
.,loc:@target_q_network/target_q_network/out/b*
_output_shapes
:*
T0
┬
target_q_network/MatMul_2MatMultarget_q_network/Tanh_1,target_q_network/target_q_network/out/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Ш
target_q_network/add_2Addtarget_q_network/MatMul_2,target_q_network/target_q_network/out/b/read*
T0*'
_output_shapes
:         
═
Dbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
valueB"   А   *
dtype0*
_output_shapes
:
┐
Bbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
valueB
 *  А┐*
dtype0
┐
Bbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
valueB
 *  А?*
dtype0
м
Lbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformDbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
seed2╩*
dtype0*
_output_shapes
:	А*

seed
к
Bbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubBbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxBbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
_output_shapes
: 
╜
Bbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulLbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformBbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w
п
>best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddBbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulBbest_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
_output_shapes
:	А
╤
#best_q_network/best_q_network/fc0/w
VariableV2*
shared_name *6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
д
*best_q_network/best_q_network/fc0/w/AssignAssign#best_q_network/best_q_network/fc0/w>best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
_output_shapes
:	А*
use_locking(*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
validate_shape(
╗
(best_q_network/best_q_network/fc0/w/readIdentity#best_q_network/best_q_network/fc0/w*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w*
_output_shapes
:	А
╝
5best_q_network/best_q_network/fc0/b/Initializer/zerosConst*6
_class,
*(loc:@best_q_network/best_q_network/fc0/b*
valueBА*    *
dtype0*
_output_shapes	
:А
╔
#best_q_network/best_q_network/fc0/b
VariableV2*
shared_name *6
_class,
*(loc:@best_q_network/best_q_network/fc0/b*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Ч
*best_q_network/best_q_network/fc0/b/AssignAssign#best_q_network/best_q_network/fc0/b5best_q_network/best_q_network/fc0/b/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/b*
validate_shape(
╖
(best_q_network/best_q_network/fc0/b/readIdentity#best_q_network/best_q_network/fc0/b*
_output_shapes	
:А*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/b
к
best_q_network/MatMulMatMulconcat(best_q_network/best_q_network/fc0/w/read*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
Н
best_q_network/addAddbest_q_network/MatMul(best_q_network/best_q_network/fc0/b/read*(
_output_shapes
:         А*
T0
░
/best_q_network/LayerNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@best_q_network/LayerNorm/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
╜
best_q_network/LayerNorm/beta
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *0
_class&
$"loc:@best_q_network/LayerNorm/beta
 
$best_q_network/LayerNorm/beta/AssignAssignbest_q_network/LayerNorm/beta/best_q_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes	
:А
е
"best_q_network/LayerNorm/beta/readIdentitybest_q_network/LayerNorm/beta*
T0*0
_class&
$"loc:@best_q_network/LayerNorm/beta*
_output_shapes	
:А
▒
/best_q_network/LayerNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@best_q_network/LayerNorm/gamma*
valueBА*  А?*
dtype0*
_output_shapes	
:А
┐
best_q_network/LayerNorm/gamma
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *1
_class'
%#loc:@best_q_network/LayerNorm/gamma
В
%best_q_network/LayerNorm/gamma/AssignAssignbest_q_network/LayerNorm/gamma/best_q_network/LayerNorm/gamma/Initializer/ones*1
_class'
%#loc:@best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
и
#best_q_network/LayerNorm/gamma/readIdentitybest_q_network/LayerNorm/gamma*
_output_shapes	
:А*
T0*1
_class'
%#loc:@best_q_network/LayerNorm/gamma
Б
7best_q_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╔
%best_q_network/LayerNorm/moments/meanMeanbest_q_network/add7best_q_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
Ц
-best_q_network/LayerNorm/moments/StopGradientStopGradient%best_q_network/LayerNorm/moments/mean*'
_output_shapes
:         *
T0
╜
2best_q_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencebest_q_network/add-best_q_network/LayerNorm/moments/StopGradient*
T0*(
_output_shapes
:         А
Е
;best_q_network/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
ё
)best_q_network/LayerNorm/moments/varianceMean2best_q_network/LayerNorm/moments/SquaredDifference;best_q_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
m
(best_q_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
┤
&best_q_network/LayerNorm/batchnorm/addAdd)best_q_network/LayerNorm/moments/variance(best_q_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
Л
(best_q_network/LayerNorm/batchnorm/RsqrtRsqrt&best_q_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
п
&best_q_network/LayerNorm/batchnorm/mulMul(best_q_network/LayerNorm/batchnorm/Rsqrt#best_q_network/LayerNorm/gamma/read*
T0*(
_output_shapes
:         А
Ю
(best_q_network/LayerNorm/batchnorm/mul_1Mulbest_q_network/add&best_q_network/LayerNorm/batchnorm/mul*(
_output_shapes
:         А*
T0
▒
(best_q_network/LayerNorm/batchnorm/mul_2Mul%best_q_network/LayerNorm/moments/mean&best_q_network/LayerNorm/batchnorm/mul*
T0*(
_output_shapes
:         А
о
&best_q_network/LayerNorm/batchnorm/subSub"best_q_network/LayerNorm/beta/read(best_q_network/LayerNorm/batchnorm/mul_2*
T0*(
_output_shapes
:         А
┤
(best_q_network/LayerNorm/batchnorm/add_1Add(best_q_network/LayerNorm/batchnorm/mul_1&best_q_network/LayerNorm/batchnorm/sub*
T0*(
_output_shapes
:         А
x
best_q_network/TanhTanh(best_q_network/LayerNorm/batchnorm/add_1*
T0*(
_output_shapes
:         А
═
Dbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
valueB"А   @   *
dtype0*
_output_shapes
:
┐
Bbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/minConst*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
┐
Bbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
valueB
 *  А?*
dtype0
м
Lbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformDbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
seed2ё*
dtype0*
_output_shapes
:	А@*

seed
к
Bbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/subSubBbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxBbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
_output_shapes
: *
T0
╜
Bbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulMulLbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformBbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes
:	А@*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w
п
>best_q_network/best_q_network/fc1/w/Initializer/random_uniformAddBbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulBbest_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
:	А@*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w
╤
#best_q_network/best_q_network/fc1/w
VariableV2*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
	container *
shape:	А@*
dtype0*
_output_shapes
:	А@*
shared_name 
д
*best_q_network/best_q_network/fc1/w/AssignAssign#best_q_network/best_q_network/fc1/w>best_q_network/best_q_network/fc1/w/Initializer/random_uniform*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@*
use_locking(
╗
(best_q_network/best_q_network/fc1/w/readIdentity#best_q_network/best_q_network/fc1/w*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
_output_shapes
:	А@
║
5best_q_network/best_q_network/fc1/b/Initializer/zerosConst*6
_class,
*(loc:@best_q_network/best_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
╟
#best_q_network/best_q_network/fc1/b
VariableV2*6
_class,
*(loc:@best_q_network/best_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ц
*best_q_network/best_q_network/fc1/b/AssignAssign#best_q_network/best_q_network/fc1/b5best_q_network/best_q_network/fc1/b/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
╢
(best_q_network/best_q_network/fc1/b/readIdentity#best_q_network/best_q_network/fc1/b*
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/b*
_output_shapes
:@
╕
best_q_network/MatMul_1MatMulbest_q_network/Tanh(best_q_network/best_q_network/fc1/w/read*
transpose_b( *
T0*'
_output_shapes
:         @*
transpose_a( 
Р
best_q_network/add_1Addbest_q_network/MatMul_1(best_q_network/best_q_network/fc1/b/read*'
_output_shapes
:         @*
T0
▓
1best_q_network/LayerNorm_1/beta/Initializer/zerosConst*2
_class(
&$loc:@best_q_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
┐
best_q_network/LayerNorm_1/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *2
_class(
&$loc:@best_q_network/LayerNorm_1/beta*
	container *
shape:@
Ж
&best_q_network/LayerNorm_1/beta/AssignAssignbest_q_network/LayerNorm_1/beta1best_q_network/LayerNorm_1/beta/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@best_q_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
к
$best_q_network/LayerNorm_1/beta/readIdentitybest_q_network/LayerNorm_1/beta*
T0*2
_class(
&$loc:@best_q_network/LayerNorm_1/beta*
_output_shapes
:@
│
1best_q_network/LayerNorm_1/gamma/Initializer/onesConst*3
_class)
'%loc:@best_q_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
┴
 best_q_network/LayerNorm_1/gamma
VariableV2*3
_class)
'%loc:@best_q_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Й
'best_q_network/LayerNorm_1/gamma/AssignAssign best_q_network/LayerNorm_1/gamma1best_q_network/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
T0*3
_class)
'%loc:@best_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
н
%best_q_network/LayerNorm_1/gamma/readIdentity best_q_network/LayerNorm_1/gamma*
_output_shapes
:@*
T0*3
_class)
'%loc:@best_q_network/LayerNorm_1/gamma
Г
9best_q_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╧
'best_q_network/LayerNorm_1/moments/meanMeanbest_q_network/add_19best_q_network/LayerNorm_1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
Ъ
/best_q_network/LayerNorm_1/moments/StopGradientStopGradient'best_q_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
┬
4best_q_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencebest_q_network/add_1/best_q_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
З
=best_q_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ў
+best_q_network/LayerNorm_1/moments/varianceMean4best_q_network/LayerNorm_1/moments/SquaredDifference=best_q_network/LayerNorm_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
o
*best_q_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
║
(best_q_network/LayerNorm_1/batchnorm/addAdd+best_q_network/LayerNorm_1/moments/variance*best_q_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
П
*best_q_network/LayerNorm_1/batchnorm/RsqrtRsqrt(best_q_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
┤
(best_q_network/LayerNorm_1/batchnorm/mulMul*best_q_network/LayerNorm_1/batchnorm/Rsqrt%best_q_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
г
*best_q_network/LayerNorm_1/batchnorm/mul_1Mulbest_q_network/add_1(best_q_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
╢
*best_q_network/LayerNorm_1/batchnorm/mul_2Mul'best_q_network/LayerNorm_1/moments/mean(best_q_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
│
(best_q_network/LayerNorm_1/batchnorm/subSub$best_q_network/LayerNorm_1/beta/read*best_q_network/LayerNorm_1/batchnorm/mul_2*'
_output_shapes
:         @*
T0
╣
*best_q_network/LayerNorm_1/batchnorm/add_1Add*best_q_network/LayerNorm_1/batchnorm/mul_1(best_q_network/LayerNorm_1/batchnorm/sub*
T0*'
_output_shapes
:         @
{
best_q_network/Tanh_1Tanh*best_q_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
═
Dbest_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
┐
Bbest_q_network/best_q_network/out/w/Initializer/random_uniform/minConst*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
valueB
 *═╠╠╜*
dtype0*
_output_shapes
: 
┐
Bbest_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
л
Lbest_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformDbest_q_network/best_q_network/out/w/Initializer/random_uniform/shape*

seed*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
seed2Ш*
dtype0*
_output_shapes

:@
к
Bbest_q_network/best_q_network/out/w/Initializer/random_uniform/subSubBbest_q_network/best_q_network/out/w/Initializer/random_uniform/maxBbest_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
_output_shapes
: 
╝
Bbest_q_network/best_q_network/out/w/Initializer/random_uniform/mulMulLbest_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformBbest_q_network/best_q_network/out/w/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
_output_shapes

:@
о
>best_q_network/best_q_network/out/w/Initializer/random_uniformAddBbest_q_network/best_q_network/out/w/Initializer/random_uniform/mulBbest_q_network/best_q_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/w
╧
#best_q_network/best_q_network/out/w
VariableV2*
shared_name *6
_class,
*(loc:@best_q_network/best_q_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
г
*best_q_network/best_q_network/out/w/AssignAssign#best_q_network/best_q_network/out/w>best_q_network/best_q_network/out/w/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
validate_shape(
║
(best_q_network/best_q_network/out/w/readIdentity#best_q_network/best_q_network/out/w*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
_output_shapes

:@
║
5best_q_network/best_q_network/out/b/Initializer/zerosConst*6
_class,
*(loc:@best_q_network/best_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
╟
#best_q_network/best_q_network/out/b
VariableV2*
shared_name *6
_class,
*(loc:@best_q_network/best_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
Ц
*best_q_network/best_q_network/out/b/AssignAssign#best_q_network/best_q_network/out/b5best_q_network/best_q_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/b
╢
(best_q_network/best_q_network/out/b/readIdentity#best_q_network/best_q_network/out/b*
_output_shapes
:*
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/b
║
best_q_network/MatMul_2MatMulbest_q_network/Tanh_1(best_q_network/best_q_network/out/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Р
best_q_network/add_2Addbest_q_network/MatMul_2(best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:         
В
SquaredDifferenceSquaredDifferencecurrent_q_network/add_2target_q_values*'
_output_shapes
:         *
T0
V
ConstConst*
_output_shapes
:*
valueB"       *
dtype0
d
MeanMeanSquaredDifferenceConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
b
q_network_loss/tagsConst*
valueB Bq_network_loss*
dtype0*
_output_shapes
: 
[
q_network_lossScalarSummaryq_network_loss/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Р
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
gradients/Mean_grad/ShapeShapeSquaredDifference*
T0*
out_type0*
_output_shapes
:
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
l
gradients/Mean_grad/Shape_1ShapeSquaredDifference*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
У
gradients/Mean_grad/ConstConst*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
╞
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0
Х
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
╩
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0
П
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1
▓
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
░
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:         *
T0
}
&gradients/SquaredDifference_grad/ShapeShapecurrent_q_network/add_2*
_output_shapes
:*
T0*
out_type0
w
(gradients/SquaredDifference_grad/Shape_1Shapetarget_q_values*
T0*
out_type0*
_output_shapes
:
▐
6gradients/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/SquaredDifference_grad/Shape(gradients/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
К
'gradients/SquaredDifference_grad/scalarConst^gradients/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
г
$gradients/SquaredDifference_grad/mulMul'gradients/SquaredDifference_grad/scalargradients/Mean_grad/truediv*
T0*'
_output_shapes
:         
е
$gradients/SquaredDifference_grad/subSubcurrent_q_network/add_2target_q_values^gradients/Mean_grad/truediv*
T0*'
_output_shapes
:         
л
&gradients/SquaredDifference_grad/mul_1Mul$gradients/SquaredDifference_grad/mul$gradients/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         
╦
$gradients/SquaredDifference_grad/SumSum&gradients/SquaredDifference_grad/mul_16gradients/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┴
(gradients/SquaredDifference_grad/ReshapeReshape$gradients/SquaredDifference_grad/Sum&gradients/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╧
&gradients/SquaredDifference_grad/Sum_1Sum&gradients/SquaredDifference_grad/mul_18gradients/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╟
*gradients/SquaredDifference_grad/Reshape_1Reshape&gradients/SquaredDifference_grad/Sum_1(gradients/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Й
$gradients/SquaredDifference_grad/NegNeg*gradients/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
Л
1gradients/SquaredDifference_grad/tuple/group_depsNoOp)^gradients/SquaredDifference_grad/Reshape%^gradients/SquaredDifference_grad/Neg
Т
9gradients/SquaredDifference_grad/tuple/control_dependencyIdentity(gradients/SquaredDifference_grad/Reshape2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/SquaredDifference_grad/Reshape*'
_output_shapes
:         
М
;gradients/SquaredDifference_grad/tuple/control_dependency_1Identity$gradients/SquaredDifference_grad/Neg2^gradients/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         *
T0*7
_class-
+)loc:@gradients/SquaredDifference_grad/Neg
Ж
,gradients/current_q_network/add_2_grad/ShapeShapecurrent_q_network/MatMul_2*
_output_shapes
:*
T0*
out_type0
x
.gradients/current_q_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
Ё
<gradients/current_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/current_q_network/add_2_grad/Shape.gradients/current_q_network/add_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ъ
*gradients/current_q_network/add_2_grad/SumSum9gradients/SquaredDifference_grad/tuple/control_dependency<gradients/current_q_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╙
.gradients/current_q_network/add_2_grad/ReshapeReshape*gradients/current_q_network/add_2_grad/Sum,gradients/current_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ю
,gradients/current_q_network/add_2_grad/Sum_1Sum9gradients/SquaredDifference_grad/tuple/control_dependency>gradients/current_q_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╠
0gradients/current_q_network/add_2_grad/Reshape_1Reshape,gradients/current_q_network/add_2_grad/Sum_1.gradients/current_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
г
7gradients/current_q_network/add_2_grad/tuple/group_depsNoOp/^gradients/current_q_network/add_2_grad/Reshape1^gradients/current_q_network/add_2_grad/Reshape_1
к
?gradients/current_q_network/add_2_grad/tuple/control_dependencyIdentity.gradients/current_q_network/add_2_grad/Reshape8^gradients/current_q_network/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/current_q_network/add_2_grad/Reshape*'
_output_shapes
:         
г
Agradients/current_q_network/add_2_grad/tuple/control_dependency_1Identity0gradients/current_q_network/add_2_grad/Reshape_18^gradients/current_q_network/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/current_q_network/add_2_grad/Reshape_1*
_output_shapes
:
Г
0gradients/current_q_network/MatMul_2_grad/MatMulMatMul?gradients/current_q_network/add_2_grad/tuple/control_dependency.current_q_network/current_q_network/out/w/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b(*
T0
ц
2gradients/current_q_network/MatMul_2_grad/MatMul_1MatMulcurrent_q_network/Tanh_1?gradients/current_q_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
к
:gradients/current_q_network/MatMul_2_grad/tuple/group_depsNoOp1^gradients/current_q_network/MatMul_2_grad/MatMul3^gradients/current_q_network/MatMul_2_grad/MatMul_1
┤
Bgradients/current_q_network/MatMul_2_grad/tuple/control_dependencyIdentity0gradients/current_q_network/MatMul_2_grad/MatMul;^gradients/current_q_network/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/current_q_network/MatMul_2_grad/MatMul*'
_output_shapes
:         @
▒
Dgradients/current_q_network/MatMul_2_grad/tuple/control_dependency_1Identity2gradients/current_q_network/MatMul_2_grad/MatMul_1;^gradients/current_q_network/MatMul_2_grad/tuple/group_deps*E
_class;
97loc:@gradients/current_q_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@*
T0
╠
0gradients/current_q_network/Tanh_1_grad/TanhGradTanhGradcurrent_q_network/Tanh_1Bgradients/current_q_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
п
Bgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/ShapeShape-current_q_network/LayerNorm_1/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
п
Dgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1Shape+current_q_network/LayerNorm_1/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
▓
Rgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/ShapeDgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Н
@gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/SumSum0gradients/current_q_network/Tanh_1_grad/TanhGradRgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Dgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeReshape@gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/SumBgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
С
Bgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Sum0gradients/current_q_network/Tanh_1_grad/TanhGradTgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ы
Fgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1ReshapeBgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Dgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
Tshape0*'
_output_shapes
:         @*
T0
х
Mgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_depsNoOpE^gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeG^gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1
В
Ugradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependencyIdentityDgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeN^gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @
И
Wgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1IdentityFgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1N^gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:         @
Щ
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeShapecurrent_q_network/add_1*
_output_shapes
:*
T0*
out_type0
п
Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1Shape+current_q_network/LayerNorm_1/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
▓
Rgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeDgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/mulMulUgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency+current_q_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
Э
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/SumSum@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/mulRgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeReshape@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/SumBgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
ы
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1Mulcurrent_q_network/add_1Ugradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
г
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1SumBgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1Tgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ы
Fgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1ReshapeBgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
х
Mgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_depsNoOpE^gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeG^gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1
В
Ugradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyIdentityDgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeN^gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:         @
И
Wgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityFgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1N^gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*Y
_classO
MKloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1
К
@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:
п
Bgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Shape_1Shape-current_q_network/LayerNorm_1/batchnorm/mul_2*
_output_shapes
:*
T0*
out_type0
м
Pgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/ShapeBgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
░
>gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/SumSumWgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Pgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
В
Bgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/ReshapeReshape>gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Sum@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:@
┤
@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Sum_1SumWgradients/current_q_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Rgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
к
>gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/NegNeg@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
У
Dgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1Reshape>gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/NegBgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
▀
Kgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_depsNoOpC^gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/ReshapeE^gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1
э
Sgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependencyIdentityBgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/ReshapeL^gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Reshape*
_output_shapes
:@
А
Ugradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1IdentityDgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1L^gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @
м
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeShape*current_q_network/LayerNorm_1/moments/mean*
T0*
out_type0*
_output_shapes
:
п
Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1Shape+current_q_network/LayerNorm_1/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
▓
Rgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeDgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/mulMulUgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1+current_q_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
Э
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/SumSum@gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/mulRgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Х
Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeReshape@gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/SumBgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
■
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1Mul*current_q_network/LayerNorm_1/moments/meanUgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:         @*
T0
г
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1SumBgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1Tgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ы
Fgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1ReshapeBgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
х
Mgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_depsNoOpE^gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeG^gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1
В
Ugradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyIdentityDgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeN^gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:         
И
Wgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityFgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1N^gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:         @
▐
gradients/AddNAddNWgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1Wgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:         @
н
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/ShapeShape-current_q_network/LayerNorm_1/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
М
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
м
Pgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/ShapeBgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
▒
>gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/mulMulgradients/AddN(current_q_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
Ч
>gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/SumSum>gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/mulPgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
П
Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/ReshapeReshape>gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Sum@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╕
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/mul_1Mul-current_q_network/LayerNorm_1/batchnorm/Rsqrtgradients/AddN*'
_output_shapes
:         @*
T0
Э
@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Sum_1Sum@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/mul_1Rgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
И
Dgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1Reshape@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Sum_1Bgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
▀
Kgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_depsNoOpC^gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/ReshapeE^gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1
·
Sgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependencyIdentityBgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/ReshapeL^gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Reshape*'
_output_shapes
:         
є
Ugradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1IdentityDgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1L^gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1*
_output_shapes
:@
Й
Fgradients/current_q_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad-current_q_network/LayerNorm_1/batchnorm/RsqrtSgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
о
@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/ShapeShape.current_q_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
Е
Bgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
Pgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/ShapeBgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Я
>gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/SumSumFgradients/current_q_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradPgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
П
Bgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/ReshapeReshape>gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Sum@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
г
@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Sum_1SumFgradients/current_q_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradRgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Д
Dgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Reshape_1Reshape@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Sum_1Bgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
▀
Kgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/tuple/group_depsNoOpC^gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/ReshapeE^gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Reshape_1
·
Sgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyIdentityBgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/ReshapeL^gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*U
_classK
IGloc:@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Reshape*'
_output_shapes
:         *
T0
я
Ugradients/current_q_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependency_1IdentityDgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Reshape_1L^gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
║
Cgradients/current_q_network/LayerNorm_1/moments/variance_grad/ShapeShape7current_q_network/LayerNorm_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
▄
Bgradients/current_q_network/LayerNorm_1/moments/variance_grad/SizeConst*
value	B :*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
╦
Agradients/current_q_network/LayerNorm_1/moments/variance_grad/addAdd@current_q_network/LayerNorm_1/moments/variance/reduction_indicesBgradients/current_q_network/LayerNorm_1/moments/variance_grad/Size*
_output_shapes
:*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape
╤
Agradients/current_q_network/LayerNorm_1/moments/variance_grad/modFloorModAgradients/current_q_network/LayerNorm_1/moments/variance_grad/addBgradients/current_q_network/LayerNorm_1/moments/variance_grad/Size*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
ч
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_1Const*
valueB:*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
у
Igradients/current_q_network/LayerNorm_1/moments/variance_grad/range/startConst*
value	B : *V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
у
Igradients/current_q_network/LayerNorm_1/moments/variance_grad/range/deltaConst*
value	B :*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
ж
Cgradients/current_q_network/LayerNorm_1/moments/variance_grad/rangeRangeIgradients/current_q_network/LayerNorm_1/moments/variance_grad/range/startBgradients/current_q_network/LayerNorm_1/moments/variance_grad/SizeIgradients/current_q_network/LayerNorm_1/moments/variance_grad/range/delta*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
т
Hgradients/current_q_network/LayerNorm_1/moments/variance_grad/Fill/valueConst*
value	B :*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
╪
Bgradients/current_q_network/LayerNorm_1/moments/variance_grad/FillFillEgradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_1Hgradients/current_q_network/LayerNorm_1/moments/variance_grad/Fill/value*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
№
Kgradients/current_q_network/LayerNorm_1/moments/variance_grad/DynamicStitchDynamicStitchCgradients/current_q_network/LayerNorm_1/moments/variance_grad/rangeAgradients/current_q_network/LayerNorm_1/moments/variance_grad/modCgradients/current_q_network/LayerNorm_1/moments/variance_grad/ShapeBgradients/current_q_network/LayerNorm_1/moments/variance_grad/Fill*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
N*#
_output_shapes
:         
с
Ggradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum/yConst*
value	B :*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
ь
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/MaximumMaximumKgradients/current_q_network/LayerNorm_1/moments/variance_grad/DynamicStitchGgradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum/y*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*#
_output_shapes
:         *
T0
█
Fgradients/current_q_network/LayerNorm_1/moments/variance_grad/floordivFloorDivCgradients/current_q_network/LayerNorm_1/moments/variance_grad/ShapeEgradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
г
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/ReshapeReshapeSgradients/current_q_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyKgradients/current_q_network/LayerNorm_1/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
ж
Bgradients/current_q_network/LayerNorm_1/moments/variance_grad/TileTileEgradients/current_q_network/LayerNorm_1/moments/variance_grad/ReshapeFgradients/current_q_network/LayerNorm_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
╝
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2Shape7current_q_network/LayerNorm_1/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
│
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_3Shape.current_q_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
ч
Cgradients/current_q_network/LayerNorm_1/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2
ю
Bgradients/current_q_network/LayerNorm_1/moments/variance_grad/ProdProdEgradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2Cgradients/current_q_network/LayerNorm_1/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2
щ
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0
Є
Dgradients/current_q_network/LayerNorm_1/moments/variance_grad/Prod_1ProdEgradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_3Egradients/current_q_network/LayerNorm_1/moments/variance_grad/Const_1*
T0*X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
х
Igradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum_1/yConst*
value	B :*X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
▐
Ggradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum_1MaximumDgradients/current_q_network/LayerNorm_1/moments/variance_grad/Prod_1Igradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum_1/y*
T0*X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
▄
Hgradients/current_q_network/LayerNorm_1/moments/variance_grad/floordiv_1FloorDivBgradients/current_q_network/LayerNorm_1/moments/variance_grad/ProdGgradients/current_q_network/LayerNorm_1/moments/variance_grad/Maximum_1*
T0*X
_classN
LJloc:@gradients/current_q_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
─
Bgradients/current_q_network/LayerNorm_1/moments/variance_grad/CastCastHgradients/current_q_network/LayerNorm_1/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
К
Egradients/current_q_network/LayerNorm_1/moments/variance_grad/truedivRealDivBgradients/current_q_network/LayerNorm_1/moments/variance_grad/TileBgradients/current_q_network/LayerNorm_1/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
г
Lgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeShapecurrent_q_network/add_1*
T0*
out_type0*
_output_shapes
:
└
Ngradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1Shape2current_q_network/LayerNorm_1/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
╨
\gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeNgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┌
Mgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/scalarConstF^gradients/current_q_network/LayerNorm_1/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Щ
Jgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/mulMulMgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/scalarEgradients/current_q_network/LayerNorm_1/moments/variance_grad/truediv*'
_output_shapes
:         @*
T0
Ш
Jgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/subSubcurrent_q_network/add_12current_q_network/LayerNorm_1/moments/StopGradientF^gradients/current_q_network/LayerNorm_1/moments/variance_grad/truediv*'
_output_shapes
:         @*
T0
Э
Lgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1MulJgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/mulJgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         @
╜
Jgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/SumSumLgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1\gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
│
Ngradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeReshapeJgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/SumLgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
┴
Lgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1SumLgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1^gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╣
Pgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1ReshapeLgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1Ngradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
╒
Jgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/NegNegPgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:         *
T0
¤
Wgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_depsNoOpO^gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeK^gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Neg
к
_gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyIdentityNgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeX^gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*a
_classW
USloc:@gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape
д
agradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityJgradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/NegX^gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         *
T0*]
_classS
QOloc:@gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/Neg
Ц
?gradients/current_q_network/LayerNorm_1/moments/mean_grad/ShapeShapecurrent_q_network/add_1*
T0*
out_type0*
_output_shapes
:
╘
>gradients/current_q_network/LayerNorm_1/moments/mean_grad/SizeConst*
value	B :*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╗
=gradients/current_q_network/LayerNorm_1/moments/mean_grad/addAdd<current_q_network/LayerNorm_1/moments/mean/reduction_indices>gradients/current_q_network/LayerNorm_1/moments/mean_grad/Size*
_output_shapes
:*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape
┴
=gradients/current_q_network/LayerNorm_1/moments/mean_grad/modFloorMod=gradients/current_q_network/LayerNorm_1/moments/mean_grad/add>gradients/current_q_network/LayerNorm_1/moments/mean_grad/Size*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
▀
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_1Const*
valueB:*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
█
Egradients/current_q_network/LayerNorm_1/moments/mean_grad/range/startConst*
_output_shapes
: *
value	B : *R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0
█
Egradients/current_q_network/LayerNorm_1/moments/mean_grad/range/deltaConst*
value	B :*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
Т
?gradients/current_q_network/LayerNorm_1/moments/mean_grad/rangeRangeEgradients/current_q_network/LayerNorm_1/moments/mean_grad/range/start>gradients/current_q_network/LayerNorm_1/moments/mean_grad/SizeEgradients/current_q_network/LayerNorm_1/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape
┌
Dgradients/current_q_network/LayerNorm_1/moments/mean_grad/Fill/valueConst*
value	B :*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╚
>gradients/current_q_network/LayerNorm_1/moments/mean_grad/FillFillAgradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_1Dgradients/current_q_network/LayerNorm_1/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape
ф
Ggradients/current_q_network/LayerNorm_1/moments/mean_grad/DynamicStitchDynamicStitch?gradients/current_q_network/LayerNorm_1/moments/mean_grad/range=gradients/current_q_network/LayerNorm_1/moments/mean_grad/mod?gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape>gradients/current_q_network/LayerNorm_1/moments/mean_grad/Fill*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
N*#
_output_shapes
:         
┘
Cgradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum/yConst*
value	B :*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
▄
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/MaximumMaximumGgradients/current_q_network/LayerNorm_1/moments/mean_grad/DynamicStitchCgradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum/y*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*#
_output_shapes
:         *
T0
╦
Bgradients/current_q_network/LayerNorm_1/moments/mean_grad/floordivFloorDiv?gradients/current_q_network/LayerNorm_1/moments/mean_grad/ShapeAgradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum*R
_classH
FDloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:*
T0
Э
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/ReshapeReshapeUgradients/current_q_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyGgradients/current_q_network/LayerNorm_1/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Ъ
>gradients/current_q_network/LayerNorm_1/moments/mean_grad/TileTileAgradients/current_q_network/LayerNorm_1/moments/mean_grad/ReshapeBgradients/current_q_network/LayerNorm_1/moments/mean_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
Ш
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2Shapecurrent_q_network/add_1*
T0*
out_type0*
_output_shapes
:
л
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_3Shape*current_q_network/LayerNorm_1/moments/mean*
_output_shapes
:*
T0*
out_type0
▀
?gradients/current_q_network/LayerNorm_1/moments/mean_grad/ConstConst*
valueB: *T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
▐
>gradients/current_q_network/LayerNorm_1/moments/mean_grad/ProdProdAgradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2?gradients/current_q_network/LayerNorm_1/moments/mean_grad/Const*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
с
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/Const_1Const*
valueB: *T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
т
@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Prod_1ProdAgradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_3Agradients/current_q_network/LayerNorm_1/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2
▌
Egradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum_1/yConst*
value	B :*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
╬
Cgradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum_1Maximum@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Prod_1Egradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum_1/y*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
╠
Dgradients/current_q_network/LayerNorm_1/moments/mean_grad/floordiv_1FloorDiv>gradients/current_q_network/LayerNorm_1/moments/mean_grad/ProdCgradients/current_q_network/LayerNorm_1/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm_1/moments/mean_grad/Shape_2
╝
>gradients/current_q_network/LayerNorm_1/moments/mean_grad/CastCastDgradients/current_q_network/LayerNorm_1/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
■
Agradients/current_q_network/LayerNorm_1/moments/mean_grad/truedivRealDiv>gradients/current_q_network/LayerNorm_1/moments/mean_grad/Tile>gradients/current_q_network/LayerNorm_1/moments/mean_grad/Cast*
T0*'
_output_shapes
:         @
з
gradients/AddN_1AddNUgradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_gradients/current_q_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyAgradients/current_q_network/LayerNorm_1/moments/mean_grad/truediv*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:         @
Ж
,gradients/current_q_network/add_1_grad/ShapeShapecurrent_q_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
x
.gradients/current_q_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
Ё
<gradients/current_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/current_q_network/add_1_grad/Shape.gradients/current_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┴
*gradients/current_q_network/add_1_grad/SumSumgradients/AddN_1<gradients/current_q_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╙
.gradients/current_q_network/add_1_grad/ReshapeReshape*gradients/current_q_network/add_1_grad/Sum,gradients/current_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
┼
,gradients/current_q_network/add_1_grad/Sum_1Sumgradients/AddN_1>gradients/current_q_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╠
0gradients/current_q_network/add_1_grad/Reshape_1Reshape,gradients/current_q_network/add_1_grad/Sum_1.gradients/current_q_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
г
7gradients/current_q_network/add_1_grad/tuple/group_depsNoOp/^gradients/current_q_network/add_1_grad/Reshape1^gradients/current_q_network/add_1_grad/Reshape_1
к
?gradients/current_q_network/add_1_grad/tuple/control_dependencyIdentity.gradients/current_q_network/add_1_grad/Reshape8^gradients/current_q_network/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/current_q_network/add_1_grad/Reshape*'
_output_shapes
:         @
г
Agradients/current_q_network/add_1_grad/tuple/control_dependency_1Identity0gradients/current_q_network/add_1_grad/Reshape_18^gradients/current_q_network/add_1_grad/tuple/group_deps*C
_class9
75loc:@gradients/current_q_network/add_1_grad/Reshape_1*
_output_shapes
:@*
T0
Д
0gradients/current_q_network/MatMul_1_grad/MatMulMatMul?gradients/current_q_network/add_1_grad/tuple/control_dependency.current_q_network/current_q_network/fc1/w/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
х
2gradients/current_q_network/MatMul_1_grad/MatMul_1MatMulcurrent_q_network/Tanh?gradients/current_q_network/add_1_grad/tuple/control_dependency*
_output_shapes
:	А@*
transpose_a(*
transpose_b( *
T0
к
:gradients/current_q_network/MatMul_1_grad/tuple/group_depsNoOp1^gradients/current_q_network/MatMul_1_grad/MatMul3^gradients/current_q_network/MatMul_1_grad/MatMul_1
╡
Bgradients/current_q_network/MatMul_1_grad/tuple/control_dependencyIdentity0gradients/current_q_network/MatMul_1_grad/MatMul;^gradients/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/current_q_network/MatMul_1_grad/MatMul*(
_output_shapes
:         А
▓
Dgradients/current_q_network/MatMul_1_grad/tuple/control_dependency_1Identity2gradients/current_q_network/MatMul_1_grad/MatMul_1;^gradients/current_q_network/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	А@*
T0*E
_class;
97loc:@gradients/current_q_network/MatMul_1_grad/MatMul_1
╔
.gradients/current_q_network/Tanh_grad/TanhGradTanhGradcurrent_q_network/TanhBgradients/current_q_network/MatMul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
л
@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/ShapeShape+current_q_network/LayerNorm/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
л
Bgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape)current_q_network/LayerNorm/batchnorm/sub*
_output_shapes
:*
T0*
out_type0
м
Pgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/ShapeBgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
З
>gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/SumSum.gradients/current_q_network/Tanh_grad/TanhGradPgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Р
Bgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshape>gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Sum@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         А
Л
@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1Sum.gradients/current_q_network/Tanh_grad/TanhGradRgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ц
Dgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1Reshape@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Sum_1Bgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         А
▀
Kgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpC^gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeE^gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
√
Sgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityBgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Reshape
Б
Ugradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityDgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1L^gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
Х
@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapecurrent_q_network/add*
T0*
out_type0*
_output_shapes
:
л
Bgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape)current_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
м
Pgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/ShapeBgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
°
>gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/mulMulSgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency)current_q_network/LayerNorm/batchnorm/mul*(
_output_shapes
:         А*
T0
Ч
>gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/SumSum>gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/mulPgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Р
Bgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshape>gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Sum@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         А
ц
@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1Mulcurrent_q_network/addSgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
Э
@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1Sum@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/mul_1Rgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ц
Dgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1Reshape@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Sum_1Bgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*(
_output_shapes
:         А*
T0*
Tshape0
▀
Kgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpC^gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeE^gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
√
Sgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityBgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         А
Б
Ugradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityDgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1L^gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
Й
>gradients/current_q_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:А*
dtype0*
_output_shapes
:
л
@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape+current_q_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
ж
Ngradients/current_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Shape@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
к
<gradients/current_q_network/LayerNorm/batchnorm/sub_grad/SumSumUgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Ngradients/current_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¤
@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/ReshapeReshape<gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Sum>gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes	
:А*
T0*
Tshape0
о
>gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Sum_1SumUgradients/current_q_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Pgradients/current_q_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ж
<gradients/current_q_network/LayerNorm/batchnorm/sub_grad/NegNeg>gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
О
Bgradients/current_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1Reshape<gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Neg@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Shape_1*(
_output_shapes
:         А*
T0*
Tshape0
┘
Igradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpA^gradients/current_q_network/LayerNorm/batchnorm/sub_grad/ReshapeC^gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1
ц
Qgradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentity@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^gradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes	
:А
∙
Sgradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityBgradients/current_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1J^gradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/sub_grad/Reshape_1*(
_output_shapes
:         А
и
@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape(current_q_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
л
Bgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape)current_q_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
м
Pgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/ShapeBgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
°
>gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/mulMulSgradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1)current_q_network/LayerNorm/batchnorm/mul*
T0*(
_output_shapes
:         А
Ч
>gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/SumSum>gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/mulPgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
П
Bgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshape>gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Sum@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
∙
@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul(current_q_network/LayerNorm/moments/meanSgradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Э
@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1Sum@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/mul_1Rgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ц
Dgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1Reshape@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Sum_1Bgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*(
_output_shapes
:         А*
T0*
Tshape0
▀
Kgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpC^gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeE^gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
·
Sgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityBgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:         
Б
Ugradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityDgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1L^gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
█
gradients/AddN_2AddNUgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1Ugradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
й
>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/ShapeShape+current_q_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
Л
@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
ж
Ngradients/current_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Shape@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
░
<gradients/current_q_network/LayerNorm/batchnorm/mul_grad/mulMulgradients/AddN_2&current_q_network/LayerNorm/gamma/read*
T0*(
_output_shapes
:         А
С
<gradients/current_q_network/LayerNorm/batchnorm/mul_grad/SumSum<gradients/current_q_network/LayerNorm/batchnorm/mul_grad/mulNgradients/current_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Й
@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/ReshapeReshape<gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Sum>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
╖
>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/mul_1Mul+current_q_network/LayerNorm/batchnorm/Rsqrtgradients/AddN_2*(
_output_shapes
:         А*
T0
Ч
>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Sum_1Sum>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/mul_1Pgradients/current_q_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
Bgradients/current_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1Reshape>gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Sum_1@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
┘
Igradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpA^gradients/current_q_network/LayerNorm/batchnorm/mul_grad/ReshapeC^gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1
Є
Qgradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentity@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^gradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:         *
T0
ь
Sgradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityBgradients/current_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1J^gradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes	
:А
Г
Dgradients/current_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad+current_q_network/LayerNorm/batchnorm/RsqrtQgradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
к
>gradients/current_q_network/LayerNorm/batchnorm/add_grad/ShapeShape,current_q_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
Г
@gradients/current_q_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ж
Ngradients/current_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/current_q_network/LayerNorm/batchnorm/add_grad/Shape@gradients/current_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Щ
<gradients/current_q_network/LayerNorm/batchnorm/add_grad/SumSumDgradients/current_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradNgradients/current_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Й
@gradients/current_q_network/LayerNorm/batchnorm/add_grad/ReshapeReshape<gradients/current_q_network/LayerNorm/batchnorm/add_grad/Sum>gradients/current_q_network/LayerNorm/batchnorm/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Э
>gradients/current_q_network/LayerNorm/batchnorm/add_grad/Sum_1SumDgradients/current_q_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradPgradients/current_q_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
■
Bgradients/current_q_network/LayerNorm/batchnorm/add_grad/Reshape_1Reshape>gradients/current_q_network/LayerNorm/batchnorm/add_grad/Sum_1@gradients/current_q_network/LayerNorm/batchnorm/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
┘
Igradients/current_q_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpA^gradients/current_q_network/LayerNorm/batchnorm/add_grad/ReshapeC^gradients/current_q_network/LayerNorm/batchnorm/add_grad/Reshape_1
Є
Qgradients/current_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentity@gradients/current_q_network/LayerNorm/batchnorm/add_grad/ReshapeJ^gradients/current_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*S
_classI
GEloc:@gradients/current_q_network/LayerNorm/batchnorm/add_grad/Reshape
ч
Sgradients/current_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityBgradients/current_q_network/LayerNorm/batchnorm/add_grad/Reshape_1J^gradients/current_q_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: *
T0
╢
Agradients/current_q_network/LayerNorm/moments/variance_grad/ShapeShape5current_q_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
╪
@gradients/current_q_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
├
?gradients/current_q_network/LayerNorm/moments/variance_grad/addAdd>current_q_network/LayerNorm/moments/variance/reduction_indices@gradients/current_q_network/LayerNorm/moments/variance_grad/Size*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
╔
?gradients/current_q_network/LayerNorm/moments/variance_grad/modFloorMod?gradients/current_q_network/LayerNorm/moments/variance_grad/add@gradients/current_q_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape
у
Cgradients/current_q_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
▀
Ggradients/current_q_network/LayerNorm/moments/variance_grad/range/startConst*
_output_shapes
: *
value	B : *T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
dtype0
▀
Ggradients/current_q_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
Ь
Agradients/current_q_network/LayerNorm/moments/variance_grad/rangeRangeGgradients/current_q_network/LayerNorm/moments/variance_grad/range/start@gradients/current_q_network/LayerNorm/moments/variance_grad/SizeGgradients/current_q_network/LayerNorm/moments/variance_grad/range/delta*

Tidx0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
▐
Fgradients/current_q_network/LayerNorm/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape
╨
@gradients/current_q_network/LayerNorm/moments/variance_grad/FillFillCgradients/current_q_network/LayerNorm/moments/variance_grad/Shape_1Fgradients/current_q_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape
Ё
Igradients/current_q_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchAgradients/current_q_network/LayerNorm/moments/variance_grad/range?gradients/current_q_network/LayerNorm/moments/variance_grad/modAgradients/current_q_network/LayerNorm/moments/variance_grad/Shape@gradients/current_q_network/LayerNorm/moments/variance_grad/Fill*
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
N*#
_output_shapes
:         
▌
Egradients/current_q_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
ф
Cgradients/current_q_network/LayerNorm/moments/variance_grad/MaximumMaximumIgradients/current_q_network/LayerNorm/moments/variance_grad/DynamicStitchEgradients/current_q_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:         *
T0*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape
╙
Dgradients/current_q_network/LayerNorm/moments/variance_grad/floordivFloorDivAgradients/current_q_network/LayerNorm/moments/variance_grad/ShapeCgradients/current_q_network/LayerNorm/moments/variance_grad/Maximum*T
_classJ
HFloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:*
T0
Э
Cgradients/current_q_network/LayerNorm/moments/variance_grad/ReshapeReshapeQgradients/current_q_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIgradients/current_q_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
а
@gradients/current_q_network/LayerNorm/moments/variance_grad/TileTileCgradients/current_q_network/LayerNorm/moments/variance_grad/ReshapeDgradients/current_q_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
╕
Cgradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2Shape5current_q_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
п
Cgradients/current_q_network/LayerNorm/moments/variance_grad/Shape_3Shape,current_q_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
у
Agradients/current_q_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
ц
@gradients/current_q_network/LayerNorm/moments/variance_grad/ProdProdCgradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2Agradients/current_q_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2
х
Cgradients/current_q_network/LayerNorm/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2
ъ
Bgradients/current_q_network/LayerNorm/moments/variance_grad/Prod_1ProdCgradients/current_q_network/LayerNorm/moments/variance_grad/Shape_3Cgradients/current_q_network/LayerNorm/moments/variance_grad/Const_1*
	keep_dims( *

Tidx0*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
с
Ggradients/current_q_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
╓
Egradients/current_q_network/LayerNorm/moments/variance_grad/Maximum_1MaximumBgradients/current_q_network/LayerNorm/moments/variance_grad/Prod_1Ggradients/current_q_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
╘
Fgradients/current_q_network/LayerNorm/moments/variance_grad/floordiv_1FloorDiv@gradients/current_q_network/LayerNorm/moments/variance_grad/ProdEgradients/current_q_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*V
_classL
JHloc:@gradients/current_q_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
└
@gradients/current_q_network/LayerNorm/moments/variance_grad/CastCastFgradients/current_q_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Е
Cgradients/current_q_network/LayerNorm/moments/variance_grad/truedivRealDiv@gradients/current_q_network/LayerNorm/moments/variance_grad/Tile@gradients/current_q_network/LayerNorm/moments/variance_grad/Cast*(
_output_shapes
:         А*
T0
Я
Jgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapecurrent_q_network/add*
T0*
out_type0*
_output_shapes
:
╝
Lgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape0current_q_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
╩
Zgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/ShapeLgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╓
Kgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/scalarConstD^gradients/current_q_network/LayerNorm/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
Ф
Hgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/mulMulKgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/scalarCgradients/current_q_network/LayerNorm/moments/variance_grad/truediv*(
_output_shapes
:         А*
T0
С
Hgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/subSubcurrent_q_network/add0current_q_network/LayerNorm/moments/StopGradientD^gradients/current_q_network/LayerNorm/moments/variance_grad/truediv*(
_output_shapes
:         А*
T0
Ш
Jgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulHgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/mulHgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*(
_output_shapes
:         А
╖
Hgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/SumSumJgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1Zgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
о
Lgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeHgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/SumJgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         А
╗
Jgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumJgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/mul_1\gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
│
Ngradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeJgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Sum_1Lgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
╤
Hgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/NegNegNgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
ў
Ugradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpM^gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeI^gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Neg
г
]gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityLgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/ReshapeV^gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Reshape*(
_output_shapes
:         А
Ь
_gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityHgradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/NegV^gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:         *
T0
Т
=gradients/current_q_network/LayerNorm/moments/mean_grad/ShapeShapecurrent_q_network/add*
T0*
out_type0*
_output_shapes
:
╨
<gradients/current_q_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
│
;gradients/current_q_network/LayerNorm/moments/mean_grad/addAdd:current_q_network/LayerNorm/moments/mean/reduction_indices<gradients/current_q_network/LayerNorm/moments/mean_grad/Size*
T0*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
╣
;gradients/current_q_network/LayerNorm/moments/mean_grad/modFloorMod;gradients/current_q_network/LayerNorm/moments/mean_grad/add<gradients/current_q_network/LayerNorm/moments/mean_grad/Size*
T0*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
█
?gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape
╫
Cgradients/current_q_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╫
Cgradients/current_q_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
И
=gradients/current_q_network/LayerNorm/moments/mean_grad/rangeRangeCgradients/current_q_network/LayerNorm/moments/mean_grad/range/start<gradients/current_q_network/LayerNorm/moments/mean_grad/SizeCgradients/current_q_network/LayerNorm/moments/mean_grad/range/delta*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
╓
Bgradients/current_q_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
└
<gradients/current_q_network/LayerNorm/moments/mean_grad/FillFill?gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_1Bgradients/current_q_network/LayerNorm/moments/mean_grad/Fill/value*
T0*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
╪
Egradients/current_q_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitch=gradients/current_q_network/LayerNorm/moments/mean_grad/range;gradients/current_q_network/LayerNorm/moments/mean_grad/mod=gradients/current_q_network/LayerNorm/moments/mean_grad/Shape<gradients/current_q_network/LayerNorm/moments/mean_grad/Fill*
T0*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
N*#
_output_shapes
:         
╒
Agradients/current_q_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╘
?gradients/current_q_network/LayerNorm/moments/mean_grad/MaximumMaximumEgradients/current_q_network/LayerNorm/moments/mean_grad/DynamicStitchAgradients/current_q_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*#
_output_shapes
:         
├
@gradients/current_q_network/LayerNorm/moments/mean_grad/floordivFloorDiv=gradients/current_q_network/LayerNorm/moments/mean_grad/Shape?gradients/current_q_network/LayerNorm/moments/mean_grad/Maximum*
T0*P
_classF
DBloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
Ч
?gradients/current_q_network/LayerNorm/moments/mean_grad/ReshapeReshapeSgradients/current_q_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyEgradients/current_q_network/LayerNorm/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Ф
<gradients/current_q_network/LayerNorm/moments/mean_grad/TileTile?gradients/current_q_network/LayerNorm/moments/mean_grad/Reshape@gradients/current_q_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
Ф
?gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2Shapecurrent_q_network/add*
T0*
out_type0*
_output_shapes
:
з
?gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_3Shape(current_q_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
█
=gradients/current_q_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
╓
<gradients/current_q_network/LayerNorm/moments/mean_grad/ProdProd?gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2=gradients/current_q_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2
▌
?gradients/current_q_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
┌
>gradients/current_q_network/LayerNorm/moments/mean_grad/Prod_1Prod?gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_3?gradients/current_q_network/LayerNorm/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2
┘
Cgradients/current_q_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
╞
Agradients/current_q_network/LayerNorm/moments/mean_grad/Maximum_1Maximum>gradients/current_q_network/LayerNorm/moments/mean_grad/Prod_1Cgradients/current_q_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2
─
Bgradients/current_q_network/LayerNorm/moments/mean_grad/floordiv_1FloorDiv<gradients/current_q_network/LayerNorm/moments/mean_grad/ProdAgradients/current_q_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*R
_classH
FDloc:@gradients/current_q_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
╕
<gradients/current_q_network/LayerNorm/moments/mean_grad/CastCastBgradients/current_q_network/LayerNorm/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
∙
?gradients/current_q_network/LayerNorm/moments/mean_grad/truedivRealDiv<gradients/current_q_network/LayerNorm/moments/mean_grad/Tile<gradients/current_q_network/LayerNorm/moments/mean_grad/Cast*
T0*(
_output_shapes
:         А
а
gradients/AddN_3AddNSgradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency]gradients/current_q_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency?gradients/current_q_network/LayerNorm/moments/mean_grad/truediv*
T0*U
_classK
IGloc:@gradients/current_q_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*(
_output_shapes
:         А
В
*gradients/current_q_network/add_grad/ShapeShapecurrent_q_network/MatMul*
T0*
out_type0*
_output_shapes
:
w
,gradients/current_q_network/add_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
ъ
:gradients/current_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/current_q_network/add_grad/Shape,gradients/current_q_network/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╜
(gradients/current_q_network/add_grad/SumSumgradients/AddN_3:gradients/current_q_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╬
,gradients/current_q_network/add_grad/ReshapeReshape(gradients/current_q_network/add_grad/Sum*gradients/current_q_network/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         А
┴
*gradients/current_q_network/add_grad/Sum_1Sumgradients/AddN_3<gradients/current_q_network/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╟
.gradients/current_q_network/add_grad/Reshape_1Reshape*gradients/current_q_network/add_grad/Sum_1,gradients/current_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
Э
5gradients/current_q_network/add_grad/tuple/group_depsNoOp-^gradients/current_q_network/add_grad/Reshape/^gradients/current_q_network/add_grad/Reshape_1
г
=gradients/current_q_network/add_grad/tuple/control_dependencyIdentity,gradients/current_q_network/add_grad/Reshape6^gradients/current_q_network/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*?
_class5
31loc:@gradients/current_q_network/add_grad/Reshape
Ь
?gradients/current_q_network/add_grad/tuple/control_dependency_1Identity.gradients/current_q_network/add_grad/Reshape_16^gradients/current_q_network/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/current_q_network/add_grad/Reshape_1*
_output_shapes	
:А
 
.gradients/current_q_network/MatMul_grad/MatMulMatMul=gradients/current_q_network/add_grad/tuple/control_dependency.current_q_network/current_q_network/fc0/w/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
╤
0gradients/current_q_network/MatMul_grad/MatMul_1MatMulconcat=gradients/current_q_network/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	А*
transpose_a(
д
8gradients/current_q_network/MatMul_grad/tuple/group_depsNoOp/^gradients/current_q_network/MatMul_grad/MatMul1^gradients/current_q_network/MatMul_grad/MatMul_1
м
@gradients/current_q_network/MatMul_grad/tuple/control_dependencyIdentity.gradients/current_q_network/MatMul_grad/MatMul9^gradients/current_q_network/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/current_q_network/MatMul_grad/MatMul*'
_output_shapes
:         
к
Bgradients/current_q_network/MatMul_grad/tuple/control_dependency_1Identity0gradients/current_q_network/MatMul_grad/MatMul_19^gradients/current_q_network/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/current_q_network/MatMul_grad/MatMul_1*
_output_shapes
:	А
У
beta1_power/initial_valueConst*
valueB
 *fff?*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
д
beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *3
_class)
'%loc:@current_q_network/LayerNorm/beta
├
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(

beta1_power/readIdentitybeta1_power*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
_output_shapes
: *
T0
У
beta2_power/initial_valueConst*
valueB
 *w╛?*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
д
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *3
_class)
'%loc:@current_q_network/LayerNorm/beta
├
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(

beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta
╒
@current_q_network/current_q_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	А*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
valueB	А*    
т
.current_q_network/current_q_network/fc0/w/Adam
VariableV2*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
┬
5current_q_network/current_q_network/fc0/w/Adam/AssignAssign.current_q_network/current_q_network/fc0/w/Adam@current_q_network/current_q_network/fc0/w/Adam/Initializer/zeros*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
╫
3current_q_network/current_q_network/fc0/w/Adam/readIdentity.current_q_network/current_q_network/fc0/w/Adam*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
_output_shapes
:	А
╫
Bcurrent_q_network/current_q_network/fc0/w/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
valueB	А*    *
dtype0*
_output_shapes
:	А
ф
0current_q_network/current_q_network/fc0/w/Adam_1
VariableV2*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
╚
7current_q_network/current_q_network/fc0/w/Adam_1/AssignAssign0current_q_network/current_q_network/fc0/w/Adam_1Bcurrent_q_network/current_q_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes
:	А
█
5current_q_network/current_q_network/fc0/w/Adam_1/readIdentity0current_q_network/current_q_network/fc0/w/Adam_1*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
_output_shapes
:	А*
T0
═
@current_q_network/current_q_network/fc0/b/Adam/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
valueBА*    *
dtype0*
_output_shapes	
:А
┌
.current_q_network/current_q_network/fc0/b/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
	container *
shape:А
╛
5current_q_network/current_q_network/fc0/b/Adam/AssignAssign.current_q_network/current_q_network/fc0/b/Adam@current_q_network/current_q_network/fc0/b/Adam/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
validate_shape(
╙
3current_q_network/current_q_network/fc0/b/Adam/readIdentity.current_q_network/current_q_network/fc0/b/Adam*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
_output_shapes	
:А
╧
Bcurrent_q_network/current_q_network/fc0/b/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
valueBА*    *
dtype0*
_output_shapes	
:А
▄
0current_q_network/current_q_network/fc0/b/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
	container 
─
7current_q_network/current_q_network/fc0/b/Adam_1/AssignAssign0current_q_network/current_q_network/fc0/b/Adam_1Bcurrent_q_network/current_q_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes	
:А
╫
5current_q_network/current_q_network/fc0/b/Adam_1/readIdentity0current_q_network/current_q_network/fc0/b/Adam_1*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
_output_shapes	
:А
╗
7current_q_network/LayerNorm/beta/Adam/Initializer/zerosConst*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
╚
%current_q_network/LayerNorm/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *3
_class)
'%loc:@current_q_network/LayerNorm/beta*
	container *
shape:А
Ъ
,current_q_network/LayerNorm/beta/Adam/AssignAssign%current_q_network/LayerNorm/beta/Adam7current_q_network/LayerNorm/beta/Adam/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(
╕
*current_q_network/LayerNorm/beta/Adam/readIdentity%current_q_network/LayerNorm/beta/Adam*
_output_shapes	
:А*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta
╜
9current_q_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
╩
'current_q_network/LayerNorm/beta/Adam_1
VariableV2*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
а
.current_q_network/LayerNorm/beta/Adam_1/AssignAssign'current_q_network/LayerNorm/beta/Adam_19current_q_network/LayerNorm/beta/Adam_1/Initializer/zeros*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
╝
,current_q_network/LayerNorm/beta/Adam_1/readIdentity'current_q_network/LayerNorm/beta/Adam_1*
_output_shapes	
:А*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta
╜
8current_q_network/LayerNorm/gamma/Adam/Initializer/zerosConst*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
╩
&current_q_network/LayerNorm/gamma/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
	container 
Ю
-current_q_network/LayerNorm/gamma/Adam/AssignAssign&current_q_network/LayerNorm/gamma/Adam8current_q_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
╗
+current_q_network/LayerNorm/gamma/Adam/readIdentity&current_q_network/LayerNorm/gamma/Adam*
_output_shapes	
:А*
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma
┐
:current_q_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
_output_shapes	
:А*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
valueBА*    *
dtype0
╠
(current_q_network/LayerNorm/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
	container *
shape:А
д
/current_q_network/LayerNorm/gamma/Adam_1/AssignAssign(current_q_network/LayerNorm/gamma/Adam_1:current_q_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
┐
-current_q_network/LayerNorm/gamma/Adam_1/readIdentity(current_q_network/LayerNorm/gamma/Adam_1*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
_output_shapes	
:А*
T0
╒
@current_q_network/current_q_network/fc1/w/Adam/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
valueB	А@*    *
dtype0*
_output_shapes
:	А@
т
.current_q_network/current_q_network/fc1/w/Adam
VariableV2*
dtype0*
_output_shapes
:	А@*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
	container *
shape:	А@
┬
5current_q_network/current_q_network/fc1/w/Adam/AssignAssign.current_q_network/current_q_network/fc1/w/Adam@current_q_network/current_q_network/fc1/w/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@
╫
3current_q_network/current_q_network/fc1/w/Adam/readIdentity.current_q_network/current_q_network/fc1/w/Adam*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
_output_shapes
:	А@
╫
Bcurrent_q_network/current_q_network/fc1/w/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
valueB	А@*    *
dtype0*
_output_shapes
:	А@
ф
0current_q_network/current_q_network/fc1/w/Adam_1
VariableV2*
	container *
shape:	А@*
dtype0*
_output_shapes
:	А@*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc1/w
╚
7current_q_network/current_q_network/fc1/w/Adam_1/AssignAssign0current_q_network/current_q_network/fc1/w/Adam_1Bcurrent_q_network/current_q_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@
█
5current_q_network/current_q_network/fc1/w/Adam_1/readIdentity0current_q_network/current_q_network/fc1/w/Adam_1*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
_output_shapes
:	А@
╦
@current_q_network/current_q_network/fc1/b/Adam/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
╪
.current_q_network/current_q_network/fc1/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
	container *
shape:@
╜
5current_q_network/current_q_network/fc1/b/Adam/AssignAssign.current_q_network/current_q_network/fc1/b/Adam@current_q_network/current_q_network/fc1/b/Adam/Initializer/zeros*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
╥
3current_q_network/current_q_network/fc1/b/Adam/readIdentity.current_q_network/current_q_network/fc1/b/Adam*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
_output_shapes
:@
═
Bcurrent_q_network/current_q_network/fc1/b/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
┌
0current_q_network/current_q_network/fc1/b/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/fc1/b
├
7current_q_network/current_q_network/fc1/b/Adam_1/AssignAssign0current_q_network/current_q_network/fc1/b/Adam_1Bcurrent_q_network/current_q_network/fc1/b/Adam_1/Initializer/zeros*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
╓
5current_q_network/current_q_network/fc1/b/Adam_1/readIdentity0current_q_network/current_q_network/fc1/b/Adam_1*
_output_shapes
:@*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b
╜
9current_q_network/LayerNorm_1/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
valueB@*    
╩
'current_q_network/LayerNorm_1/beta/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
	container *
shape:@
б
.current_q_network/LayerNorm_1/beta/Adam/AssignAssign'current_q_network/LayerNorm_1/beta/Adam9current_q_network/LayerNorm_1/beta/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
╜
,current_q_network/LayerNorm_1/beta/Adam/readIdentity'current_q_network/LayerNorm_1/beta/Adam*
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
_output_shapes
:@
┐
;current_q_network/LayerNorm_1/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
valueB@*    
╠
)current_q_network/LayerNorm_1/beta/Adam_1
VariableV2*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
з
0current_q_network/LayerNorm_1/beta/Adam_1/AssignAssign)current_q_network/LayerNorm_1/beta/Adam_1;current_q_network/LayerNorm_1/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
┴
.current_q_network/LayerNorm_1/beta/Adam_1/readIdentity)current_q_network/LayerNorm_1/beta/Adam_1*
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
_output_shapes
:@
┐
:current_q_network/LayerNorm_1/gamma/Adam/Initializer/zerosConst*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╠
(current_q_network/LayerNorm_1/gamma/Adam
VariableV2*
shared_name *6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
е
/current_q_network/LayerNorm_1/gamma/Adam/AssignAssign(current_q_network/LayerNorm_1/gamma/Adam:current_q_network/LayerNorm_1/gamma/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
validate_shape(
└
-current_q_network/LayerNorm_1/gamma/Adam/readIdentity(current_q_network/LayerNorm_1/gamma/Adam*
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
_output_shapes
:@
┴
<current_q_network/LayerNorm_1/gamma/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╬
*current_q_network/LayerNorm_1/gamma/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
л
1current_q_network/LayerNorm_1/gamma/Adam_1/AssignAssign*current_q_network/LayerNorm_1/gamma/Adam_1<current_q_network/LayerNorm_1/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
─
/current_q_network/LayerNorm_1/gamma/Adam_1/readIdentity*current_q_network/LayerNorm_1/gamma/Adam_1*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
_output_shapes
:@*
T0
╙
@current_q_network/current_q_network/out/w/Adam/Initializer/zerosConst*
_output_shapes

:@*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
valueB@*    *
dtype0
р
.current_q_network/current_q_network/out/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/out/w*
	container *
shape
:@
┴
5current_q_network/current_q_network/out/w/Adam/AssignAssign.current_q_network/current_q_network/out/w/Adam@current_q_network/current_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
╓
3current_q_network/current_q_network/out/w/Adam/readIdentity.current_q_network/current_q_network/out/w/Adam*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
_output_shapes

:@
╒
Bcurrent_q_network/current_q_network/out/w/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
т
0current_q_network/current_q_network/out/w/Adam_1
VariableV2*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
╟
7current_q_network/current_q_network/out/w/Adam_1/AssignAssign0current_q_network/current_q_network/out/w/Adam_1Bcurrent_q_network/current_q_network/out/w/Adam_1/Initializer/zeros*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
┌
5current_q_network/current_q_network/out/w/Adam_1/readIdentity0current_q_network/current_q_network/out/w/Adam_1*
_output_shapes

:@*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w
╦
@current_q_network/current_q_network/out/b/Adam/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
╪
.current_q_network/current_q_network/out/b/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/out/b
╜
5current_q_network/current_q_network/out/b/Adam/AssignAssign.current_q_network/current_q_network/out/b/Adam@current_q_network/current_q_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
╥
3current_q_network/current_q_network/out/b/Adam/readIdentity.current_q_network/current_q_network/out/b/Adam*
_output_shapes
:*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b
═
Bcurrent_q_network/current_q_network/out/b/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
┌
0current_q_network/current_q_network/out/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *<
_class2
0.loc:@current_q_network/current_q_network/out/b
├
7current_q_network/current_q_network/out/b/Adam_1/AssignAssign0current_q_network/current_q_network/out/b/Adam_1Bcurrent_q_network/current_q_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
validate_shape(
╓
5current_q_network/current_q_network/out/b/Adam_1/readIdentity0current_q_network/current_q_network/out/b/Adam_1*
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
_output_shapes
:
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
Е
?Adam/update_current_q_network/current_q_network/fc0/w/ApplyAdam	ApplyAdam)current_q_network/current_q_network/fc0/w.current_q_network/current_q_network/fc0/w/Adam0current_q_network/current_q_network/fc0/w/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/current_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
use_nesterov( *
_output_shapes
:	А
■
?Adam/update_current_q_network/current_q_network/fc0/b/ApplyAdam	ApplyAdam)current_q_network/current_q_network/fc0/b.current_q_network/current_q_network/fc0/b/Adam0current_q_network/current_q_network/fc0/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/current_q_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
use_nesterov( *
_output_shapes	
:А
у
6Adam/update_current_q_network/LayerNorm/beta/ApplyAdam	ApplyAdam current_q_network/LayerNorm/beta%current_q_network/LayerNorm/beta/Adam'current_q_network/LayerNorm/beta/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/current_q_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
use_nesterov( *
_output_shapes	
:А
ъ
7Adam/update_current_q_network/LayerNorm/gamma/ApplyAdam	ApplyAdam!current_q_network/LayerNorm/gamma&current_q_network/LayerNorm/gamma/Adam(current_q_network/LayerNorm/gamma/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/current_q_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes	
:А
З
?Adam/update_current_q_network/current_q_network/fc1/w/ApplyAdam	ApplyAdam)current_q_network/current_q_network/fc1/w.current_q_network/current_q_network/fc1/w/Adam0current_q_network/current_q_network/fc1/w/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/current_q_network/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes
:	А@*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
use_nesterov( 
 
?Adam/update_current_q_network/current_q_network/fc1/b/ApplyAdam	ApplyAdam)current_q_network/current_q_network/fc1/b.current_q_network/current_q_network/fc1/b/Adam0current_q_network/current_q_network/fc1/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/current_q_network/add_1_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
use_nesterov( *
_output_shapes
:@*
use_locking( 
ю
8Adam/update_current_q_network/LayerNorm_1/beta/ApplyAdam	ApplyAdam"current_q_network/LayerNorm_1/beta'current_q_network/LayerNorm_1/beta/Adam)current_q_network/LayerNorm_1/beta/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/current_q_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta*
use_nesterov( *
_output_shapes
:@
ї
9Adam/update_current_q_network/LayerNorm_1/gamma/ApplyAdam	ApplyAdam#current_q_network/LayerNorm_1/gamma(current_q_network/LayerNorm_1/gamma/Adam*current_q_network/LayerNorm_1/gamma/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonUgradients/current_q_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
use_nesterov( *
_output_shapes
:@
Ж
?Adam/update_current_q_network/current_q_network/out/w/ApplyAdam	ApplyAdam)current_q_network/current_q_network/out/w.current_q_network/current_q_network/out/w/Adam0current_q_network/current_q_network/out/w/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/current_q_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w*
use_nesterov( *
_output_shapes

:@
 
?Adam/update_current_q_network/current_q_network/out/b/ApplyAdam	ApplyAdam)current_q_network/current_q_network/out/b.current_q_network/current_q_network/out/b/Adam0current_q_network/current_q_network/out/b/Adam_1beta1_power/readbeta2_power/readlearning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/current_q_network/add_2_grad/tuple/control_dependency_1*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
∙
Adam/mulMulbeta1_power/read
Adam/beta1@^Adam/update_current_q_network/current_q_network/fc0/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc0/b/ApplyAdam7^Adam/update_current_q_network/LayerNorm/beta/ApplyAdam8^Adam/update_current_q_network/LayerNorm/gamma/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc1/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc1/b/ApplyAdam9^Adam/update_current_q_network/LayerNorm_1/beta/ApplyAdam:^Adam/update_current_q_network/LayerNorm_1/gamma/ApplyAdam@^Adam/update_current_q_network/current_q_network/out/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/out/b/ApplyAdam*
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
_output_shapes
: 
л
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
√

Adam/mul_1Mulbeta2_power/read
Adam/beta2@^Adam/update_current_q_network/current_q_network/fc0/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc0/b/ApplyAdam7^Adam/update_current_q_network/LayerNorm/beta/ApplyAdam8^Adam/update_current_q_network/LayerNorm/gamma/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc1/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc1/b/ApplyAdam9^Adam/update_current_q_network/LayerNorm_1/beta/ApplyAdam:^Adam/update_current_q_network/LayerNorm_1/gamma/ApplyAdam@^Adam/update_current_q_network/current_q_network/out/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/out/b/ApplyAdam*
_output_shapes
: *
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta
п
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
а
AdamNoOp@^Adam/update_current_q_network/current_q_network/fc0/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc0/b/ApplyAdam7^Adam/update_current_q_network/LayerNorm/beta/ApplyAdam8^Adam/update_current_q_network/LayerNorm/gamma/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc1/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/fc1/b/ApplyAdam9^Adam/update_current_q_network/LayerNorm_1/beta/ApplyAdam:^Adam/update_current_q_network/LayerNorm_1/gamma/ApplyAdam@^Adam/update_current_q_network/current_q_network/out/w/ApplyAdam@^Adam/update_current_q_network/current_q_network/out/b/ApplyAdam^Adam/Assign^Adam/Assign_1
█
AssignAssigntarget_q_network/LayerNorm/beta%current_q_network/LayerNorm/beta/read*
use_locking( *
T0*2
_class(
&$loc:@target_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes	
:А
р
Assign_1Assign target_q_network/LayerNorm/gamma&current_q_network/LayerNorm/gamma/read*
use_locking( *
T0*3
_class)
'%loc:@target_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
т
Assign_2Assign!target_q_network/LayerNorm_1/beta'current_q_network/LayerNorm_1/beta/read*
_output_shapes
:@*
use_locking( *
T0*4
_class*
(&loc:@target_q_network/LayerNorm_1/beta*
validate_shape(
х
Assign_3Assign"target_q_network/LayerNorm_1/gamma(current_q_network/LayerNorm_1/gamma/read*5
_class+
)'loc:@target_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
Ў
Assign_4Assign'target_q_network/target_q_network/fc0/b.current_q_network/current_q_network/fc0/b/read*
_output_shapes	
:А*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/b*
validate_shape(
·
Assign_5Assign'target_q_network/target_q_network/fc0/w.current_q_network/current_q_network/fc0/w/read*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
validate_shape(*
_output_shapes
:	А
ї
Assign_6Assign'target_q_network/target_q_network/fc1/b.current_q_network/current_q_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/b
·
Assign_7Assign'target_q_network/target_q_network/fc1/w.current_q_network/current_q_network/fc1/w/read*
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@*
use_locking( 
ї
Assign_8Assign'target_q_network/target_q_network/out/b.current_q_network/current_q_network/out/b/read*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/b*
validate_shape(*
_output_shapes
:
∙
Assign_9Assign'target_q_network/target_q_network/out/w.current_q_network/current_q_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
validate_shape(
~

group_depsNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
┘
	Assign_10Assignbest_q_network/LayerNorm/beta$target_q_network/LayerNorm/beta/read*
use_locking( *
T0*0
_class&
$"loc:@best_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes	
:А
▄
	Assign_11Assignbest_q_network/LayerNorm/gamma%target_q_network/LayerNorm/gamma/read*
use_locking( *
T0*1
_class'
%#loc:@best_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
▐
	Assign_12Assignbest_q_network/LayerNorm_1/beta&target_q_network/LayerNorm_1/beta/read*
T0*2
_class(
&$loc:@best_q_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking( 
с
	Assign_13Assign best_q_network/LayerNorm_1/gamma'target_q_network/LayerNorm_1/gamma/read*
T0*3
_class)
'%loc:@best_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( 
э
	Assign_14Assign#best_q_network/best_q_network/fc0/b,target_q_network/target_q_network/fc0/b/read*
_output_shapes	
:А*
use_locking( *
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/b*
validate_shape(
ё
	Assign_15Assign#best_q_network/best_q_network/fc0/w,target_q_network/target_q_network/fc0/w/read*
validate_shape(*
_output_shapes
:	А*
use_locking( *
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc0/w
ь
	Assign_16Assign#best_q_network/best_q_network/fc1/b,target_q_network/target_q_network/fc1/b/read*
_output_shapes
:@*
use_locking( *
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/b*
validate_shape(
ё
	Assign_17Assign#best_q_network/best_q_network/fc1/w,target_q_network/target_q_network/fc1/w/read*
use_locking( *
T0*6
_class,
*(loc:@best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes
:	А@
ь
	Assign_18Assign#best_q_network/best_q_network/out/b,target_q_network/target_q_network/out/b/read*
use_locking( *
T0*6
_class,
*(loc:@best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:
Ё
	Assign_19Assign#best_q_network/best_q_network/out/w,target_q_network/target_q_network/out/w/read*6
_class,
*(loc:@best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
М
group_deps_1NoOp
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
█
	Assign_20Assigntarget_q_network/LayerNorm/beta"best_q_network/LayerNorm/beta/read*
_output_shapes	
:А*
use_locking( *
T0*2
_class(
&$loc:@target_q_network/LayerNorm/beta*
validate_shape(
▐
	Assign_21Assign target_q_network/LayerNorm/gamma#best_q_network/LayerNorm/gamma/read*3
_class)
'%loc:@target_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking( *
T0
р
	Assign_22Assign!target_q_network/LayerNorm_1/beta$best_q_network/LayerNorm_1/beta/read*
use_locking( *
T0*4
_class*
(&loc:@target_q_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
у
	Assign_23Assign"target_q_network/LayerNorm_1/gamma%best_q_network/LayerNorm_1/gamma/read*5
_class+
)'loc:@target_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
ё
	Assign_24Assign'target_q_network/target_q_network/fc0/b(best_q_network/best_q_network/fc0/b/read*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/b*
validate_shape(*
_output_shapes	
:А
ї
	Assign_25Assign'target_q_network/target_q_network/fc0/w(best_q_network/best_q_network/fc0/w/read*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc0/w*
validate_shape(*
_output_shapes
:	А
Ё
	Assign_26Assign'target_q_network/target_q_network/fc1/b(best_q_network/best_q_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/b
ї
	Assign_27Assign'target_q_network/target_q_network/fc1/w(best_q_network/best_q_network/fc1/w/read*
_output_shapes
:	А@*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/fc1/w*
validate_shape(
Ё
	Assign_28Assign'target_q_network/target_q_network/out/b(best_q_network/best_q_network/out/b/read*
use_locking( *
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/b*
validate_shape(*
_output_shapes
:
Ї
	Assign_29Assign'target_q_network/target_q_network/out/w(best_q_network/best_q_network/out/w/read*
T0*:
_class0
.,loc:@target_q_network/target_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( 
▌
	Assign_30Assign current_q_network/LayerNorm/beta"best_q_network/LayerNorm/beta/read*
use_locking( *
T0*3
_class)
'%loc:@current_q_network/LayerNorm/beta*
validate_shape(*
_output_shapes	
:А
р
	Assign_31Assign!current_q_network/LayerNorm/gamma#best_q_network/LayerNorm/gamma/read*
use_locking( *
T0*4
_class*
(&loc:@current_q_network/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:А
т
	Assign_32Assign"current_q_network/LayerNorm_1/beta$best_q_network/LayerNorm_1/beta/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*5
_class+
)'loc:@current_q_network/LayerNorm_1/beta
х
	Assign_33Assign#current_q_network/LayerNorm_1/gamma%best_q_network/LayerNorm_1/gamma/read*
use_locking( *
T0*6
_class,
*(loc:@current_q_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
ї
	Assign_34Assign)current_q_network/current_q_network/fc0/b(best_q_network/best_q_network/fc0/b/read*
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes	
:А*
use_locking( 
∙
	Assign_35Assign)current_q_network/current_q_network/fc0/w(best_q_network/best_q_network/fc0/w/read*
_output_shapes
:	А*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc0/w*
validate_shape(
Ї
	Assign_36Assign)current_q_network/current_q_network/fc1/b(best_q_network/best_q_network/fc1/b/read*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
∙
	Assign_37Assign)current_q_network/current_q_network/fc1/w(best_q_network/best_q_network/fc1/w/read*
_output_shapes
:	А@*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/fc1/w*
validate_shape(
Ї
	Assign_38Assign)current_q_network/current_q_network/out/b(best_q_network/best_q_network/out/b/read*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
°
	Assign_39Assign)current_q_network/current_q_network/out/w(best_q_network/best_q_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*<
_class2
0.loc:@current_q_network/current_q_network/out/w
Д
group_deps_2NoOp
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
S
Merge/MergeSummaryMergeSummaryq_network_loss*
N*
_output_shapes
: 
╬
initNoOp1^current_q_network/current_q_network/fc0/w/Assign1^current_q_network/current_q_network/fc0/b/Assign(^current_q_network/LayerNorm/beta/Assign)^current_q_network/LayerNorm/gamma/Assign1^current_q_network/current_q_network/fc1/w/Assign1^current_q_network/current_q_network/fc1/b/Assign*^current_q_network/LayerNorm_1/beta/Assign+^current_q_network/LayerNorm_1/gamma/Assign1^current_q_network/current_q_network/out/w/Assign1^current_q_network/current_q_network/out/b/Assign/^target_q_network/target_q_network/fc0/w/Assign/^target_q_network/target_q_network/fc0/b/Assign'^target_q_network/LayerNorm/beta/Assign(^target_q_network/LayerNorm/gamma/Assign/^target_q_network/target_q_network/fc1/w/Assign/^target_q_network/target_q_network/fc1/b/Assign)^target_q_network/LayerNorm_1/beta/Assign*^target_q_network/LayerNorm_1/gamma/Assign/^target_q_network/target_q_network/out/w/Assign/^target_q_network/target_q_network/out/b/Assign+^best_q_network/best_q_network/fc0/w/Assign+^best_q_network/best_q_network/fc0/b/Assign%^best_q_network/LayerNorm/beta/Assign&^best_q_network/LayerNorm/gamma/Assign+^best_q_network/best_q_network/fc1/w/Assign+^best_q_network/best_q_network/fc1/b/Assign'^best_q_network/LayerNorm_1/beta/Assign(^best_q_network/LayerNorm_1/gamma/Assign+^best_q_network/best_q_network/out/w/Assign+^best_q_network/best_q_network/out/b/Assign^beta1_power/Assign^beta2_power/Assign6^current_q_network/current_q_network/fc0/w/Adam/Assign8^current_q_network/current_q_network/fc0/w/Adam_1/Assign6^current_q_network/current_q_network/fc0/b/Adam/Assign8^current_q_network/current_q_network/fc0/b/Adam_1/Assign-^current_q_network/LayerNorm/beta/Adam/Assign/^current_q_network/LayerNorm/beta/Adam_1/Assign.^current_q_network/LayerNorm/gamma/Adam/Assign0^current_q_network/LayerNorm/gamma/Adam_1/Assign6^current_q_network/current_q_network/fc1/w/Adam/Assign8^current_q_network/current_q_network/fc1/w/Adam_1/Assign6^current_q_network/current_q_network/fc1/b/Adam/Assign8^current_q_network/current_q_network/fc1/b/Adam_1/Assign/^current_q_network/LayerNorm_1/beta/Adam/Assign1^current_q_network/LayerNorm_1/beta/Adam_1/Assign0^current_q_network/LayerNorm_1/gamma/Adam/Assign2^current_q_network/LayerNorm_1/gamma/Adam_1/Assign6^current_q_network/current_q_network/out/w/Adam/Assign8^current_q_network/current_q_network/out/w/Adam_1/Assign6^current_q_network/current_q_network/out/b/Adam/Assign8^current_q_network/current_q_network/out/b/Adam_1/Assign
T
learning_rate_1Placeholder*
_output_shapes
:*
shape:*
dtype0
q
observations_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
x
target_observations_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
j
returnsPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
щ
Rcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
█
Pcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
█
Pcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╒
Zcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/shape*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
seed2о*
dtype0*
_output_shapes

:@*

seed*
T0
т
Pcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/subSubPcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxPcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
_output_shapes
: 
Ї
Pcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulMulZcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformPcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
_output_shapes

:@
ц
Lcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniformAddPcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulPcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
_output_shapes

:@*
T0
ы
1current_value_network/current_value_network/fc0/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
	container 
█
8current_value_network/current_value_network/fc0/w/AssignAssign1current_value_network/current_value_network/fc0/wLcurrent_value_network/current_value_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
ф
6current_value_network/current_value_network/fc0/w/readIdentity1current_value_network/current_value_network/fc0/w*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
_output_shapes

:@*
T0
╓
Ccurrent_value_network/current_value_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
valueB@*    
у
1current_value_network/current_value_network/fc0/b
VariableV2*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
╬
8current_value_network/current_value_network/fc0/b/AssignAssign1current_value_network/current_value_network/fc0/bCcurrent_value_network/current_value_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
validate_shape(
р
6current_value_network/current_value_network/fc0/b/readIdentity1current_value_network/current_value_network/fc0/b*
_output_shapes
:@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b
╞
current_value_network/MatMulMatMulobservations_16current_value_network/current_value_network/fc0/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
и
current_value_network/addAddcurrent_value_network/MatMul6current_value_network/current_value_network/fc0/b/read*'
_output_shapes
:         @*
T0
╝
6current_value_network/LayerNorm/beta/Initializer/zerosConst*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╔
$current_value_network/LayerNorm/beta
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@current_value_network/LayerNorm/beta
Ъ
+current_value_network/LayerNorm/beta/AssignAssign$current_value_network/LayerNorm/beta6current_value_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta
╣
)current_value_network/LayerNorm/beta/readIdentity$current_value_network/LayerNorm/beta*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
_output_shapes
:@
╜
6current_value_network/LayerNorm/gamma/Initializer/onesConst*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╦
%current_value_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
	container *
shape:@
Э
,current_value_network/LayerNorm/gamma/AssignAssign%current_value_network/LayerNorm/gamma6current_value_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
╝
*current_value_network/LayerNorm/gamma/readIdentity%current_value_network/LayerNorm/gamma*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
_output_shapes
:@*
T0
И
>current_value_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
▐
,current_value_network/LayerNorm/moments/meanMeancurrent_value_network/add>current_value_network/LayerNorm/moments/mean/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
д
4current_value_network/LayerNorm/moments/StopGradientStopGradient,current_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:         
╤
9current_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencecurrent_value_network/add4current_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:         @
М
Bcurrent_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ж
0current_value_network/LayerNorm/moments/varianceMean9current_value_network/LayerNorm/moments/SquaredDifferenceBcurrent_value_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
t
/current_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╔
-current_value_network/LayerNorm/batchnorm/addAdd0current_value_network/LayerNorm/moments/variance/current_value_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:         *
T0
Щ
/current_value_network/LayerNorm/batchnorm/RsqrtRsqrt-current_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
├
-current_value_network/LayerNorm/batchnorm/mulMul/current_value_network/LayerNorm/batchnorm/Rsqrt*current_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:         @
▓
/current_value_network/LayerNorm/batchnorm/mul_1Mulcurrent_value_network/add-current_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┼
/current_value_network/LayerNorm/batchnorm/mul_2Mul,current_value_network/LayerNorm/moments/mean-current_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┬
-current_value_network/LayerNorm/batchnorm/subSub)current_value_network/LayerNorm/beta/read/current_value_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╚
/current_value_network/LayerNorm/batchnorm/add_1Add/current_value_network/LayerNorm/batchnorm/mul_1-current_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:         @
Е
current_value_network/TanhTanh/current_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:         @
щ
Rcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
█
Pcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/minConst*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
█
Pcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
valueB
 *  А?*
dtype0
╒
Zcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
seed2╒
т
Pcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/subSubPcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxPcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
_output_shapes
: 
Ї
Pcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulMulZcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformPcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w
ц
Lcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniformAddPcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulPcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w
ы
1current_value_network/current_value_network/fc1/w
VariableV2*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
█
8current_value_network/current_value_network/fc1/w/AssignAssign1current_value_network/current_value_network/fc1/wLcurrent_value_network/current_value_network/fc1/w/Initializer/random_uniform*
_output_shapes

:@@*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
validate_shape(
ф
6current_value_network/current_value_network/fc1/w/readIdentity1current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w
╓
Ccurrent_value_network/current_value_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1current_value_network/current_value_network/fc1/b
VariableV2*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
╬
8current_value_network/current_value_network/fc1/b/AssignAssign1current_value_network/current_value_network/fc1/bCcurrent_value_network/current_value_network/fc1/b/Initializer/zeros*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
р
6current_value_network/current_value_network/fc1/b/readIdentity1current_value_network/current_value_network/fc1/b*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
_output_shapes
:@
╘
current_value_network/MatMul_1MatMulcurrent_value_network/Tanh6current_value_network/current_value_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
м
current_value_network/add_1Addcurrent_value_network/MatMul_16current_value_network/current_value_network/fc1/b/read*
T0*'
_output_shapes
:         @
└
8current_value_network/LayerNorm_1/beta/Initializer/zerosConst*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
═
&current_value_network/LayerNorm_1/beta
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_value_network/LayerNorm_1/beta
в
-current_value_network/LayerNorm_1/beta/AssignAssign&current_value_network/LayerNorm_1/beta8current_value_network/LayerNorm_1/beta/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
┐
+current_value_network/LayerNorm_1/beta/readIdentity&current_value_network/LayerNorm_1/beta*
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
_output_shapes
:@
┴
8current_value_network/LayerNorm_1/gamma/Initializer/onesConst*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╧
'current_value_network/LayerNorm_1/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
	container *
shape:@
е
.current_value_network/LayerNorm_1/gamma/AssignAssign'current_value_network/LayerNorm_1/gamma8current_value_network/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
┬
,current_value_network/LayerNorm_1/gamma/readIdentity'current_value_network/LayerNorm_1/gamma*
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
_output_shapes
:@
К
@current_value_network/LayerNorm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
ф
.current_value_network/LayerNorm_1/moments/meanMeancurrent_value_network/add_1@current_value_network/LayerNorm_1/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
и
6current_value_network/LayerNorm_1/moments/StopGradientStopGradient.current_value_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
╫
;current_value_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencecurrent_value_network/add_16current_value_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
О
Dcurrent_value_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
М
2current_value_network/LayerNorm_1/moments/varianceMean;current_value_network/LayerNorm_1/moments/SquaredDifferenceDcurrent_value_network/LayerNorm_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
v
1current_value_network/LayerNorm_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *╠╝М+
╧
/current_value_network/LayerNorm_1/batchnorm/addAdd2current_value_network/LayerNorm_1/moments/variance1current_value_network/LayerNorm_1/batchnorm/add/y*'
_output_shapes
:         *
T0
Э
1current_value_network/LayerNorm_1/batchnorm/RsqrtRsqrt/current_value_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
╔
/current_value_network/LayerNorm_1/batchnorm/mulMul1current_value_network/LayerNorm_1/batchnorm/Rsqrt,current_value_network/LayerNorm_1/gamma/read*'
_output_shapes
:         @*
T0
╕
1current_value_network/LayerNorm_1/batchnorm/mul_1Mulcurrent_value_network/add_1/current_value_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
╦
1current_value_network/LayerNorm_1/batchnorm/mul_2Mul.current_value_network/LayerNorm_1/moments/mean/current_value_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
╚
/current_value_network/LayerNorm_1/batchnorm/subSub+current_value_network/LayerNorm_1/beta/read1current_value_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╬
1current_value_network/LayerNorm_1/batchnorm/add_1Add1current_value_network/LayerNorm_1/batchnorm/mul_1/current_value_network/LayerNorm_1/batchnorm/sub*'
_output_shapes
:         @*
T0
Й
current_value_network/Tanh_1Tanh1current_value_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
щ
Rcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@current_value_network/current_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
█
Pcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/minConst*D
_class:
86loc:@current_value_network/current_value_network/out/w*
valueB
 *═╠╠╜*
dtype0*
_output_shapes
: 
█
Pcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@current_value_network/current_value_network/out/w*
valueB
 *═╠╠=
╒
Zcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/shape*

seed*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w*
seed2№*
dtype0*
_output_shapes

:@
т
Pcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/subSubPcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/maxPcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w
Ї
Pcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/mulMulZcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformPcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/sub*D
_class:
86loc:@current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
ц
Lcurrent_value_network/current_value_network/out/w/Initializer/random_uniformAddPcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/mulPcurrent_value_network/current_value_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w*
_output_shapes

:@
ы
1current_value_network/current_value_network/out/w
VariableV2*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
█
8current_value_network/current_value_network/out/w/AssignAssign1current_value_network/current_value_network/out/wLcurrent_value_network/current_value_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w
ф
6current_value_network/current_value_network/out/w/readIdentity1current_value_network/current_value_network/out/w*D
_class:
86loc:@current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
╓
Ccurrent_value_network/current_value_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@current_value_network/current_value_network/out/b*
valueB*    
у
1current_value_network/current_value_network/out/b
VariableV2*D
_class:
86loc:@current_value_network/current_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
╬
8current_value_network/current_value_network/out/b/AssignAssign1current_value_network/current_value_network/out/bCcurrent_value_network/current_value_network/out/b/Initializer/zeros*D
_class:
86loc:@current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
р
6current_value_network/current_value_network/out/b/readIdentity1current_value_network/current_value_network/out/b*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b*
_output_shapes
:
╓
current_value_network/MatMul_2MatMulcurrent_value_network/Tanh_16current_value_network/current_value_network/out/w/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
м
current_value_network/add_2Addcurrent_value_network/MatMul_26current_value_network/current_value_network/out/b/read*
T0*'
_output_shapes
:         
х
Ptarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
╫
Ntarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/minConst*
_output_shapes
: *B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
valueB
 *  А┐*
dtype0
╫
Ntarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/maxConst*
_output_shapes
: *B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
valueB
 *  А?*
dtype0
╧
Xtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
seed2М
┌
Ntarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/subSubNtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/maxNtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w
ь
Ntarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/mulMulXtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/RandomUniformNtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
_output_shapes

:@
▐
Jtarget_value_network/target_value_network/fc0/w/Initializer/random_uniformAddNtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/mulNtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w
ч
/target_value_network/target_value_network/fc0/w
VariableV2*
shared_name *B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
╙
6target_value_network/target_value_network/fc0/w/AssignAssign/target_value_network/target_value_network/fc0/wJtarget_value_network/target_value_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w
▐
4target_value_network/target_value_network/fc0/w/readIdentity/target_value_network/target_value_network/fc0/w*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
_output_shapes

:@
╥
Atarget_value_network/target_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@target_value_network/target_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
▀
/target_value_network/target_value_network/fc0/b
VariableV2*
shared_name *B
_class8
64loc:@target_value_network/target_value_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
╞
6target_value_network/target_value_network/fc0/b/AssignAssign/target_value_network/target_value_network/fc0/bAtarget_value_network/target_value_network/fc0/b/Initializer/zeros*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
┌
4target_value_network/target_value_network/fc0/b/readIdentity/target_value_network/target_value_network/fc0/b*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/b*
_output_shapes
:@
╩
target_value_network/MatMulMatMultarget_observations_14target_value_network/target_value_network/fc0/w/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b( *
T0
д
target_value_network/addAddtarget_value_network/MatMul4target_value_network/target_value_network/fc0/b/read*'
_output_shapes
:         @*
T0
║
5target_value_network/LayerNorm/beta/Initializer/zerosConst*6
_class,
*(loc:@target_value_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╟
#target_value_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@target_value_network/LayerNorm/beta*
	container *
shape:@
Ц
*target_value_network/LayerNorm/beta/AssignAssign#target_value_network/LayerNorm/beta5target_value_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@target_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
╢
(target_value_network/LayerNorm/beta/readIdentity#target_value_network/LayerNorm/beta*
T0*6
_class,
*(loc:@target_value_network/LayerNorm/beta*
_output_shapes
:@
╗
5target_value_network/LayerNorm/gamma/Initializer/onesConst*7
_class-
+)loc:@target_value_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╔
$target_value_network/LayerNorm/gamma
VariableV2*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@target_value_network/LayerNorm/gamma*
	container *
shape:@*
dtype0
Щ
+target_value_network/LayerNorm/gamma/AssignAssign$target_value_network/LayerNorm/gamma5target_value_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*7
_class-
+)loc:@target_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
╣
)target_value_network/LayerNorm/gamma/readIdentity$target_value_network/LayerNorm/gamma*
T0*7
_class-
+)loc:@target_value_network/LayerNorm/gamma*
_output_shapes
:@
З
=target_value_network/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
█
+target_value_network/LayerNorm/moments/meanMeantarget_value_network/add=target_value_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
в
3target_value_network/LayerNorm/moments/StopGradientStopGradient+target_value_network/LayerNorm/moments/mean*'
_output_shapes
:         *
T0
╬
8target_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencetarget_value_network/add3target_value_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:         @
Л
Atarget_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Г
/target_value_network/LayerNorm/moments/varianceMean8target_value_network/LayerNorm/moments/SquaredDifferenceAtarget_value_network/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
s
.target_value_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╞
,target_value_network/LayerNorm/batchnorm/addAdd/target_value_network/LayerNorm/moments/variance.target_value_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
Ч
.target_value_network/LayerNorm/batchnorm/RsqrtRsqrt,target_value_network/LayerNorm/batchnorm/add*'
_output_shapes
:         *
T0
└
,target_value_network/LayerNorm/batchnorm/mulMul.target_value_network/LayerNorm/batchnorm/Rsqrt)target_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:         @
п
.target_value_network/LayerNorm/batchnorm/mul_1Multarget_value_network/add,target_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┬
.target_value_network/LayerNorm/batchnorm/mul_2Mul+target_value_network/LayerNorm/moments/mean,target_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
┐
,target_value_network/LayerNorm/batchnorm/subSub(target_value_network/LayerNorm/beta/read.target_value_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:         @
┼
.target_value_network/LayerNorm/batchnorm/add_1Add.target_value_network/LayerNorm/batchnorm/mul_1,target_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:         @
Г
target_value_network/TanhTanh.target_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:         @
х
Ptarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
╫
Ntarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/minConst*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╫
Ntarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╧
Xtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformPtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
seed2│
┌
Ntarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/subSubNtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/maxNtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
_output_shapes
: 
ь
Ntarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/mulMulXtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/RandomUniformNtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
_output_shapes

:@@
▐
Jtarget_value_network/target_value_network/fc1/w/Initializer/random_uniformAddNtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/mulNtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
_output_shapes

:@@
ч
/target_value_network/target_value_network/fc1/w
VariableV2*
_output_shapes

:@@*
shared_name *B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
	container *
shape
:@@*
dtype0
╙
6target_value_network/target_value_network/fc1/w/AssignAssign/target_value_network/target_value_network/fc1/wJtarget_value_network/target_value_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
▐
4target_value_network/target_value_network/fc1/w/readIdentity/target_value_network/target_value_network/fc1/w*
_output_shapes

:@@*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w
╥
Atarget_value_network/target_value_network/fc1/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*B
_class8
64loc:@target_value_network/target_value_network/fc1/b*
valueB@*    
▀
/target_value_network/target_value_network/fc1/b
VariableV2*
shared_name *B
_class8
64loc:@target_value_network/target_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
╞
6target_value_network/target_value_network/fc1/b/AssignAssign/target_value_network/target_value_network/fc1/bAtarget_value_network/target_value_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/b*
validate_shape(
┌
4target_value_network/target_value_network/fc1/b/readIdentity/target_value_network/target_value_network/fc1/b*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/b*
_output_shapes
:@
╨
target_value_network/MatMul_1MatMultarget_value_network/Tanh4target_value_network/target_value_network/fc1/w/read*
transpose_b( *
T0*'
_output_shapes
:         @*
transpose_a( 
и
target_value_network/add_1Addtarget_value_network/MatMul_14target_value_network/target_value_network/fc1/b/read*
T0*'
_output_shapes
:         @
╛
7target_value_network/LayerNorm_1/beta/Initializer/zerosConst*8
_class.
,*loc:@target_value_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╦
%target_value_network/LayerNorm_1/beta
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@target_value_network/LayerNorm_1/beta
Ю
,target_value_network/LayerNorm_1/beta/AssignAssign%target_value_network/LayerNorm_1/beta7target_value_network/LayerNorm_1/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@target_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
╝
*target_value_network/LayerNorm_1/beta/readIdentity%target_value_network/LayerNorm_1/beta*8
_class.
,*loc:@target_value_network/LayerNorm_1/beta*
_output_shapes
:@*
T0
┐
7target_value_network/LayerNorm_1/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*9
_class/
-+loc:@target_value_network/LayerNorm_1/gamma*
valueB@*  А?
═
&target_value_network/LayerNorm_1/gamma
VariableV2*9
_class/
-+loc:@target_value_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
б
-target_value_network/LayerNorm_1/gamma/AssignAssign&target_value_network/LayerNorm_1/gamma7target_value_network/LayerNorm_1/gamma/Initializer/ones*
_output_shapes
:@*
use_locking(*
T0*9
_class/
-+loc:@target_value_network/LayerNorm_1/gamma*
validate_shape(
┐
+target_value_network/LayerNorm_1/gamma/readIdentity&target_value_network/LayerNorm_1/gamma*
T0*9
_class/
-+loc:@target_value_network/LayerNorm_1/gamma*
_output_shapes
:@
Й
?target_value_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
с
-target_value_network/LayerNorm_1/moments/meanMeantarget_value_network/add_1?target_value_network/LayerNorm_1/moments/mean/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
ж
5target_value_network/LayerNorm_1/moments/StopGradientStopGradient-target_value_network/LayerNorm_1/moments/mean*'
_output_shapes
:         *
T0
╘
:target_value_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencetarget_value_network/add_15target_value_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
Н
Ctarget_value_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Й
1target_value_network/LayerNorm_1/moments/varianceMean:target_value_network/LayerNorm_1/moments/SquaredDifferenceCtarget_value_network/LayerNorm_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
u
0target_value_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╠
.target_value_network/LayerNorm_1/batchnorm/addAdd1target_value_network/LayerNorm_1/moments/variance0target_value_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
Ы
0target_value_network/LayerNorm_1/batchnorm/RsqrtRsqrt.target_value_network/LayerNorm_1/batchnorm/add*'
_output_shapes
:         *
T0
╞
.target_value_network/LayerNorm_1/batchnorm/mulMul0target_value_network/LayerNorm_1/batchnorm/Rsqrt+target_value_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
╡
0target_value_network/LayerNorm_1/batchnorm/mul_1Multarget_value_network/add_1.target_value_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
╚
0target_value_network/LayerNorm_1/batchnorm/mul_2Mul-target_value_network/LayerNorm_1/moments/mean.target_value_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
┼
.target_value_network/LayerNorm_1/batchnorm/subSub*target_value_network/LayerNorm_1/beta/read0target_value_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╦
0target_value_network/LayerNorm_1/batchnorm/add_1Add0target_value_network/LayerNorm_1/batchnorm/mul_1.target_value_network/LayerNorm_1/batchnorm/sub*
T0*'
_output_shapes
:         @
З
target_value_network/Tanh_1Tanh0target_value_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
х
Ptarget_value_network/target_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@target_value_network/target_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
╫
Ntarget_value_network/target_value_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *B
_class8
64loc:@target_value_network/target_value_network/out/w*
valueB
 *═╠╠╜*
dtype0
╫
Ntarget_value_network/target_value_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@target_value_network/target_value_network/out/w*
valueB
 *═╠╠=
╧
Xtarget_value_network/target_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPtarget_value_network/target_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w*
seed2┌
┌
Ntarget_value_network/target_value_network/out/w/Initializer/random_uniform/subSubNtarget_value_network/target_value_network/out/w/Initializer/random_uniform/maxNtarget_value_network/target_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w
ь
Ntarget_value_network/target_value_network/out/w/Initializer/random_uniform/mulMulXtarget_value_network/target_value_network/out/w/Initializer/random_uniform/RandomUniformNtarget_value_network/target_value_network/out/w/Initializer/random_uniform/sub*B
_class8
64loc:@target_value_network/target_value_network/out/w*
_output_shapes

:@*
T0
▐
Jtarget_value_network/target_value_network/out/w/Initializer/random_uniformAddNtarget_value_network/target_value_network/out/w/Initializer/random_uniform/mulNtarget_value_network/target_value_network/out/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w*
_output_shapes

:@
ч
/target_value_network/target_value_network/out/w
VariableV2*
_output_shapes

:@*
shared_name *B
_class8
64loc:@target_value_network/target_value_network/out/w*
	container *
shape
:@*
dtype0
╙
6target_value_network/target_value_network/out/w/AssignAssign/target_value_network/target_value_network/out/wJtarget_value_network/target_value_network/out/w/Initializer/random_uniform*
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
▐
4target_value_network/target_value_network/out/w/readIdentity/target_value_network/target_value_network/out/w*
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w*
_output_shapes

:@
╥
Atarget_value_network/target_value_network/out/b/Initializer/zerosConst*B
_class8
64loc:@target_value_network/target_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
▀
/target_value_network/target_value_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *B
_class8
64loc:@target_value_network/target_value_network/out/b*
	container *
shape:
╞
6target_value_network/target_value_network/out/b/AssignAssign/target_value_network/target_value_network/out/bAtarget_value_network/target_value_network/out/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@target_value_network/target_value_network/out/b*
validate_shape(*
_output_shapes
:
┌
4target_value_network/target_value_network/out/b/readIdentity/target_value_network/target_value_network/out/b*
T0*B
_class8
64loc:@target_value_network/target_value_network/out/b*
_output_shapes
:
╥
target_value_network/MatMul_2MatMultarget_value_network/Tanh_14target_value_network/target_value_network/out/w/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
и
target_value_network/add_2Addtarget_value_network/MatMul_24target_value_network/target_value_network/out/b/read*
T0*'
_output_shapes
:         
▌
Lbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
╧
Jbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/minConst*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╧
Jbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
├
Tbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformLbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
seed2ъ*
dtype0*
_output_shapes

:@*

seed
╩
Jbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubJbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxJbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
_output_shapes
: 
▄
Jbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulTbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformJbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
_output_shapes

:@
╬
Fbest_value_network/best_value_network/fc0/w/Initializer/random_uniformAddJbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulJbest_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/w
▀
+best_value_network/best_value_network/fc0/w
VariableV2*
shared_name *>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
├
2best_value_network/best_value_network/fc0/w/AssignAssign+best_value_network/best_value_network/fc0/wFbest_value_network/best_value_network/fc0/w/Initializer/random_uniform*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
╥
0best_value_network/best_value_network/fc0/w/readIdentity+best_value_network/best_value_network/fc0/w*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
_output_shapes

:@
╩
=best_value_network/best_value_network/fc0/b/Initializer/zerosConst*>
_class4
20loc:@best_value_network/best_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
╫
+best_value_network/best_value_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *>
_class4
20loc:@best_value_network/best_value_network/fc0/b*
	container *
shape:@
╢
2best_value_network/best_value_network/fc0/b/AssignAssign+best_value_network/best_value_network/fc0/b=best_value_network/best_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/b
╬
0best_value_network/best_value_network/fc0/b/readIdentity+best_value_network/best_value_network/fc0/b*>
_class4
20loc:@best_value_network/best_value_network/fc0/b*
_output_shapes
:@*
T0
╜
best_value_network/MatMulMatMulobservations_10best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
Ь
best_value_network/addAddbest_value_network/MatMul0best_value_network/best_value_network/fc0/b/read*'
_output_shapes
:         @*
T0
╢
3best_value_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:@*4
_class*
(&loc:@best_value_network/LayerNorm/beta*
valueB@*    *
dtype0
├
!best_value_network/LayerNorm/beta
VariableV2*
shared_name *4
_class*
(&loc:@best_value_network/LayerNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
О
(best_value_network/LayerNorm/beta/AssignAssign!best_value_network/LayerNorm/beta3best_value_network/LayerNorm/beta/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*4
_class*
(&loc:@best_value_network/LayerNorm/beta*
validate_shape(
░
&best_value_network/LayerNorm/beta/readIdentity!best_value_network/LayerNorm/beta*
T0*4
_class*
(&loc:@best_value_network/LayerNorm/beta*
_output_shapes
:@
╖
3best_value_network/LayerNorm/gamma/Initializer/onesConst*5
_class+
)'loc:@best_value_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
┼
"best_value_network/LayerNorm/gamma
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *5
_class+
)'loc:@best_value_network/LayerNorm/gamma
С
)best_value_network/LayerNorm/gamma/AssignAssign"best_value_network/LayerNorm/gamma3best_value_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@best_value_network/LayerNorm/gamma*
validate_shape(
│
'best_value_network/LayerNorm/gamma/readIdentity"best_value_network/LayerNorm/gamma*
T0*5
_class+
)'loc:@best_value_network/LayerNorm/gamma*
_output_shapes
:@
Е
;best_value_network/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
╒
)best_value_network/LayerNorm/moments/meanMeanbest_value_network/add;best_value_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
Ю
1best_value_network/LayerNorm/moments/StopGradientStopGradient)best_value_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:         
╚
6best_value_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencebest_value_network/add1best_value_network/LayerNorm/moments/StopGradient*'
_output_shapes
:         @*
T0
Й
?best_value_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
¤
-best_value_network/LayerNorm/moments/varianceMean6best_value_network/LayerNorm/moments/SquaredDifference?best_value_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
q
,best_value_network/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *╠╝М+
└
*best_value_network/LayerNorm/batchnorm/addAdd-best_value_network/LayerNorm/moments/variance,best_value_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
У
,best_value_network/LayerNorm/batchnorm/RsqrtRsqrt*best_value_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
║
*best_value_network/LayerNorm/batchnorm/mulMul,best_value_network/LayerNorm/batchnorm/Rsqrt'best_value_network/LayerNorm/gamma/read*'
_output_shapes
:         @*
T0
й
,best_value_network/LayerNorm/batchnorm/mul_1Mulbest_value_network/add*best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
╝
,best_value_network/LayerNorm/batchnorm/mul_2Mul)best_value_network/LayerNorm/moments/mean*best_value_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
╣
*best_value_network/LayerNorm/batchnorm/subSub&best_value_network/LayerNorm/beta/read,best_value_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:         @
┐
,best_value_network/LayerNorm/batchnorm/add_1Add,best_value_network/LayerNorm/batchnorm/mul_1*best_value_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:         @

best_value_network/TanhTanh,best_value_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:         @
▌
Lbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/shapeConst*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
╧
Jbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/minConst*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╧
Jbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxConst*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
├
Tbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformLbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
seed2С
╩
Jbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/subSubJbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxJbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/w
▄
Jbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulMulTbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformJbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
╬
Fbest_value_network/best_value_network/fc1/w/Initializer/random_uniformAddJbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulJbest_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/w
▀
+best_value_network/best_value_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
	container *
shape
:@@
├
2best_value_network/best_value_network/fc1/w/AssignAssign+best_value_network/best_value_network/fc1/wFbest_value_network/best_value_network/fc1/w/Initializer/random_uniform*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
╥
0best_value_network/best_value_network/fc1/w/readIdentity+best_value_network/best_value_network/fc1/w*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
╩
=best_value_network/best_value_network/fc1/b/Initializer/zerosConst*>
_class4
20loc:@best_value_network/best_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
╫
+best_value_network/best_value_network/fc1/b
VariableV2*>
_class4
20loc:@best_value_network/best_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
╢
2best_value_network/best_value_network/fc1/b/AssignAssign+best_value_network/best_value_network/fc1/b=best_value_network/best_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
╬
0best_value_network/best_value_network/fc1/b/readIdentity+best_value_network/best_value_network/fc1/b*
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/b*
_output_shapes
:@
╚
best_value_network/MatMul_1MatMulbest_value_network/Tanh0best_value_network/best_value_network/fc1/w/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b( *
T0
а
best_value_network/add_1Addbest_value_network/MatMul_10best_value_network/best_value_network/fc1/b/read*'
_output_shapes
:         @*
T0
║
5best_value_network/LayerNorm_1/beta/Initializer/zerosConst*6
_class,
*(loc:@best_value_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╟
#best_value_network/LayerNorm_1/beta
VariableV2*6
_class,
*(loc:@best_value_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ц
*best_value_network/LayerNorm_1/beta/AssignAssign#best_value_network/LayerNorm_1/beta5best_value_network/LayerNorm_1/beta/Initializer/zeros*6
_class,
*(loc:@best_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
╢
(best_value_network/LayerNorm_1/beta/readIdentity#best_value_network/LayerNorm_1/beta*
T0*6
_class,
*(loc:@best_value_network/LayerNorm_1/beta*
_output_shapes
:@
╗
5best_value_network/LayerNorm_1/gamma/Initializer/onesConst*7
_class-
+)loc:@best_value_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╔
$best_value_network/LayerNorm_1/gamma
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@best_value_network/LayerNorm_1/gamma
Щ
+best_value_network/LayerNorm_1/gamma/AssignAssign$best_value_network/LayerNorm_1/gamma5best_value_network/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
T0*7
_class-
+)loc:@best_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╣
)best_value_network/LayerNorm_1/gamma/readIdentity$best_value_network/LayerNorm_1/gamma*
_output_shapes
:@*
T0*7
_class-
+)loc:@best_value_network/LayerNorm_1/gamma
З
=best_value_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
█
+best_value_network/LayerNorm_1/moments/meanMeanbest_value_network/add_1=best_value_network/LayerNorm_1/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
в
3best_value_network/LayerNorm_1/moments/StopGradientStopGradient+best_value_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
╬
8best_value_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencebest_value_network/add_13best_value_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
Л
Abest_value_network/LayerNorm_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
Г
/best_value_network/LayerNorm_1/moments/varianceMean8best_value_network/LayerNorm_1/moments/SquaredDifferenceAbest_value_network/LayerNorm_1/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
s
.best_value_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╞
,best_value_network/LayerNorm_1/batchnorm/addAdd/best_value_network/LayerNorm_1/moments/variance.best_value_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
Ч
.best_value_network/LayerNorm_1/batchnorm/RsqrtRsqrt,best_value_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
└
,best_value_network/LayerNorm_1/batchnorm/mulMul.best_value_network/LayerNorm_1/batchnorm/Rsqrt)best_value_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
п
.best_value_network/LayerNorm_1/batchnorm/mul_1Mulbest_value_network/add_1,best_value_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
┬
.best_value_network/LayerNorm_1/batchnorm/mul_2Mul+best_value_network/LayerNorm_1/moments/mean,best_value_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
┐
,best_value_network/LayerNorm_1/batchnorm/subSub(best_value_network/LayerNorm_1/beta/read.best_value_network/LayerNorm_1/batchnorm/mul_2*'
_output_shapes
:         @*
T0
┼
.best_value_network/LayerNorm_1/batchnorm/add_1Add.best_value_network/LayerNorm_1/batchnorm/mul_1,best_value_network/LayerNorm_1/batchnorm/sub*
T0*'
_output_shapes
:         @
Г
best_value_network/Tanh_1Tanh.best_value_network/LayerNorm_1/batchnorm/add_1*'
_output_shapes
:         @*
T0
▌
Lbest_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@best_value_network/best_value_network/out/w*
valueB"@      *
dtype0
╧
Jbest_value_network/best_value_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *>
_class4
20loc:@best_value_network/best_value_network/out/w*
valueB
 *═╠╠╜*
dtype0
╧
Jbest_value_network/best_value_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@best_value_network/best_value_network/out/w*
valueB
 *═╠╠=
├
Tbest_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformLbest_value_network/best_value_network/out/w/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/w*
seed2╕*
dtype0*
_output_shapes

:@*

seed
╩
Jbest_value_network/best_value_network/out/w/Initializer/random_uniform/subSubJbest_value_network/best_value_network/out/w/Initializer/random_uniform/maxJbest_value_network/best_value_network/out/w/Initializer/random_uniform/min*>
_class4
20loc:@best_value_network/best_value_network/out/w*
_output_shapes
: *
T0
▄
Jbest_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulTbest_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformJbest_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/w
╬
Fbest_value_network/best_value_network/out/w/Initializer/random_uniformAddJbest_value_network/best_value_network/out/w/Initializer/random_uniform/mulJbest_value_network/best_value_network/out/w/Initializer/random_uniform/min*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/w*
_output_shapes

:@
▀
+best_value_network/best_value_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *>
_class4
20loc:@best_value_network/best_value_network/out/w*
	container *
shape
:@
├
2best_value_network/best_value_network/out/w/AssignAssign+best_value_network/best_value_network/out/wFbest_value_network/best_value_network/out/w/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/w*
validate_shape(
╥
0best_value_network/best_value_network/out/w/readIdentity+best_value_network/best_value_network/out/w*
_output_shapes

:@*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/w
╩
=best_value_network/best_value_network/out/b/Initializer/zerosConst*>
_class4
20loc:@best_value_network/best_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
╫
+best_value_network/best_value_network/out/b
VariableV2*
shared_name *>
_class4
20loc:@best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
╢
2best_value_network/best_value_network/out/b/AssignAssign+best_value_network/best_value_network/out/b=best_value_network/best_value_network/out/b/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:
╬
0best_value_network/best_value_network/out/b/readIdentity+best_value_network/best_value_network/out/b*
T0*>
_class4
20loc:@best_value_network/best_value_network/out/b*
_output_shapes
:
╩
best_value_network/MatMul_2MatMulbest_value_network/Tanh_10best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
а
best_value_network/add_2Addbest_value_network/MatMul_20best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:         
А
SquaredDifference_1SquaredDifferencecurrent_value_network/add_2returns*'
_output_shapes
:         *
T0
X
Const_1Const*
_output_shapes
:*
valueB"       *
dtype0
j
Mean_1MeanSquaredDifference_1Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
value_network_loss/tagsConst*#
valueB Bvalue_network_loss*
dtype0*
_output_shapes
: 
e
value_network_lossScalarSummaryvalue_network_loss/tagsMean_1*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
V
gradients_1/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
_output_shapes
: *
T0
v
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ъ
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
p
gradients_1/Mean_1_grad/ShapeShapeSquaredDifference_1*
out_type0*
_output_shapes
:*
T0
и
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
r
gradients_1/Mean_1_grad/Shape_1ShapeSquaredDifference_1*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
Ы
gradients_1/Mean_1_grad/ConstConst*
_output_shapes
:*
valueB: *2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1*
dtype0
╓
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1
Э
gradients_1/Mean_1_grad/Const_1Const*
valueB: *2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1*
dtype0*
_output_shapes
:
┌
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0
Ч
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1*
dtype0*
_output_shapes
: 
┬
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1*
_output_shapes
: 
└
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0*2
_class(
&$loc:@gradients_1/Mean_1_grad/Shape_1
v
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ш
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*'
_output_shapes
:         
Е
*gradients_1/SquaredDifference_1_grad/ShapeShapecurrent_value_network/add_2*
T0*
out_type0*
_output_shapes
:
s
,gradients_1/SquaredDifference_1_grad/Shape_1Shapereturns*
T0*
out_type0*
_output_shapes
:
ъ
:gradients_1/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/SquaredDifference_1_grad/Shape,gradients_1/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Т
+gradients_1/SquaredDifference_1_grad/scalarConst ^gradients_1/Mean_1_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
п
(gradients_1/SquaredDifference_1_grad/mulMul+gradients_1/SquaredDifference_1_grad/scalargradients_1/Mean_1_grad/truediv*'
_output_shapes
:         *
T0
й
(gradients_1/SquaredDifference_1_grad/subSubcurrent_value_network/add_2returns ^gradients_1/Mean_1_grad/truediv*
T0*'
_output_shapes
:         
╖
*gradients_1/SquaredDifference_1_grad/mul_1Mul(gradients_1/SquaredDifference_1_grad/mul(gradients_1/SquaredDifference_1_grad/sub*'
_output_shapes
:         *
T0
╫
(gradients_1/SquaredDifference_1_grad/SumSum*gradients_1/SquaredDifference_1_grad/mul_1:gradients_1/SquaredDifference_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
═
,gradients_1/SquaredDifference_1_grad/ReshapeReshape(gradients_1/SquaredDifference_1_grad/Sum*gradients_1/SquaredDifference_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
█
*gradients_1/SquaredDifference_1_grad/Sum_1Sum*gradients_1/SquaredDifference_1_grad/mul_1<gradients_1/SquaredDifference_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╙
.gradients_1/SquaredDifference_1_grad/Reshape_1Reshape*gradients_1/SquaredDifference_1_grad/Sum_1,gradients_1/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
С
(gradients_1/SquaredDifference_1_grad/NegNeg.gradients_1/SquaredDifference_1_grad/Reshape_1*
T0*'
_output_shapes
:         
Ч
5gradients_1/SquaredDifference_1_grad/tuple/group_depsNoOp-^gradients_1/SquaredDifference_1_grad/Reshape)^gradients_1/SquaredDifference_1_grad/Neg
в
=gradients_1/SquaredDifference_1_grad/tuple/control_dependencyIdentity,gradients_1/SquaredDifference_1_grad/Reshape6^gradients_1/SquaredDifference_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*?
_class5
31loc:@gradients_1/SquaredDifference_1_grad/Reshape
Ь
?gradients_1/SquaredDifference_1_grad/tuple/control_dependency_1Identity(gradients_1/SquaredDifference_1_grad/Neg6^gradients_1/SquaredDifference_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/SquaredDifference_1_grad/Neg*'
_output_shapes
:         
Р
2gradients_1/current_value_network/add_2_grad/ShapeShapecurrent_value_network/MatMul_2*
out_type0*
_output_shapes
:*
T0
~
4gradients_1/current_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
В
Bgradients_1/current_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/current_value_network/add_2_grad/Shape4gradients_1/current_value_network/add_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
·
0gradients_1/current_value_network/add_2_grad/SumSum=gradients_1/SquaredDifference_1_grad/tuple/control_dependencyBgradients_1/current_value_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
4gradients_1/current_value_network/add_2_grad/ReshapeReshape0gradients_1/current_value_network/add_2_grad/Sum2gradients_1/current_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
■
2gradients_1/current_value_network/add_2_grad/Sum_1Sum=gradients_1/SquaredDifference_1_grad/tuple/control_dependencyDgradients_1/current_value_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
▐
6gradients_1/current_value_network/add_2_grad/Reshape_1Reshape2gradients_1/current_value_network/add_2_grad/Sum_14gradients_1/current_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
╡
=gradients_1/current_value_network/add_2_grad/tuple/group_depsNoOp5^gradients_1/current_value_network/add_2_grad/Reshape7^gradients_1/current_value_network/add_2_grad/Reshape_1
┬
Egradients_1/current_value_network/add_2_grad/tuple/control_dependencyIdentity4gradients_1/current_value_network/add_2_grad/Reshape>^gradients_1/current_value_network/add_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/current_value_network/add_2_grad/Reshape*'
_output_shapes
:         
╗
Ggradients_1/current_value_network/add_2_grad/tuple/control_dependency_1Identity6gradients_1/current_value_network/add_2_grad/Reshape_1>^gradients_1/current_value_network/add_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/current_value_network/add_2_grad/Reshape_1*
_output_shapes
:
Ч
6gradients_1/current_value_network/MatMul_2_grad/MatMulMatMulEgradients_1/current_value_network/add_2_grad/tuple/control_dependency6current_value_network/current_value_network/out/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
Ў
8gradients_1/current_value_network/MatMul_2_grad/MatMul_1MatMulcurrent_value_network/Tanh_1Egradients_1/current_value_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
╝
@gradients_1/current_value_network/MatMul_2_grad/tuple/group_depsNoOp7^gradients_1/current_value_network/MatMul_2_grad/MatMul9^gradients_1/current_value_network/MatMul_2_grad/MatMul_1
╠
Hgradients_1/current_value_network/MatMul_2_grad/tuple/control_dependencyIdentity6gradients_1/current_value_network/MatMul_2_grad/MatMulA^gradients_1/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/current_value_network/MatMul_2_grad/MatMul*'
_output_shapes
:         @
╔
Jgradients_1/current_value_network/MatMul_2_grad/tuple/control_dependency_1Identity8gradients_1/current_value_network/MatMul_2_grad/MatMul_1A^gradients_1/current_value_network/MatMul_2_grad/tuple/group_deps*K
_classA
?=loc:@gradients_1/current_value_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@*
T0
▄
6gradients_1/current_value_network/Tanh_1_grad/TanhGradTanhGradcurrent_value_network/Tanh_1Hgradients_1/current_value_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
╣
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/ShapeShape1current_value_network/LayerNorm_1/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
╣
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1Shape/current_value_network/LayerNorm_1/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
─
Xgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/ShapeJgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Я
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/SumSum6gradients_1/current_value_network/Tanh_1_grad/TanhGradXgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
з
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeReshapeFgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/SumHgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
г
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Sum6gradients_1/current_value_network/Tanh_1_grad/TanhGradZgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
н
Lgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1ReshapeHgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Jgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ў
Sgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_depsNoOpK^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeM^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1
Ъ
[gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependencyIdentityJgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeT^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @
а
]gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1IdentityLgradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1T^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*_
_classU
SQloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1
г
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeShapecurrent_value_network/add_1*
T0*
out_type0*
_output_shapes
:
╣
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1Shape/current_value_network/LayerNorm_1/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
─
Xgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeJgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Н
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/mulMul[gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency/current_value_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
п
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/SumSumFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/mulXgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
з
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeReshapeFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/SumHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
√
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1Mulcurrent_value_network/add_1[gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
╡
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1SumHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1Zgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
н
Lgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1ReshapeHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ў
Sgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_depsNoOpK^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeM^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1
Ъ
[gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyIdentityJgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeT^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:         @
а
]gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityLgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1T^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:         @
Р
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:
╣
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Shape_1Shape1current_value_network/LayerNorm_1/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
╛
Vgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/ShapeHgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┬
Dgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/SumSum]gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Vgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ф
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/ReshapeReshapeDgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/SumFgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:@
╞
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Sum_1Sum]gradients_1/current_value_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Xgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╢
Dgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/NegNegFgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
е
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1ReshapeDgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/NegHgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ё
Qgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_depsNoOpI^gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/ReshapeK^gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1
Е
Ygradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependencyIdentityHgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/ReshapeR^gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Reshape*
_output_shapes
:@
Ш
[gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1IdentityJgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1R^gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @*
T0
╢
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeShape.current_value_network/LayerNorm_1/moments/mean*
T0*
out_type0*
_output_shapes
:
╣
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1Shape/current_value_network/LayerNorm_1/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
─
Xgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeJgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Н
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/mulMul[gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1/current_value_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
п
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/SumSumFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/mulXgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
з
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeReshapeFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/SumHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
О
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1Mul.current_value_network/LayerNorm_1/moments/mean[gradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         @
╡
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1SumHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1Zgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
н
Lgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1ReshapeHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ў
Sgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_depsNoOpK^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeM^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1
Ъ
[gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyIdentityJgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeT^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:         
а
]gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityLgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1T^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:         @
Є
gradients_1/AddNAddN]gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1]gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:         @
╖
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/ShapeShape1current_value_network/LayerNorm_1/batchnorm/Rsqrt*
_output_shapes
:*
T0*
out_type0
Т
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
╛
Vgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/ShapeHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╜
Dgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/mulMulgradients_1/AddN,current_value_network/LayerNorm_1/gamma/read*'
_output_shapes
:         @*
T0
й
Dgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/SumSumDgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/mulVgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/ReshapeReshapeDgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/SumFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
─
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/mul_1Mul1current_value_network/LayerNorm_1/batchnorm/Rsqrtgradients_1/AddN*'
_output_shapes
:         @*
T0
п
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Sum_1SumFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/mul_1Xgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ъ
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1ReshapeFgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Sum_1Hgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
ё
Qgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_depsNoOpI^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/ReshapeK^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1
Т
Ygradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependencyIdentityHgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/ReshapeR^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Reshape*'
_output_shapes
:         
Л
[gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1IdentityJgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1R^gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1*
_output_shapes
:@
Щ
Lgradients_1/current_value_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad1current_value_network/LayerNorm_1/batchnorm/RsqrtYgradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
╕
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/ShapeShape2current_value_network/LayerNorm_1/moments/variance*
_output_shapes
:*
T0*
out_type0
Л
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╛
Vgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/ShapeHgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
▒
Dgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/SumSumLgradients_1/current_value_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradVgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Hgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/ReshapeReshapeDgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/SumFgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╡
Fgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Sum_1SumLgradients_1/current_value_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradXgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ц
Jgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Reshape_1ReshapeFgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Sum_1Hgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ё
Qgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/tuple/group_depsNoOpI^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/ReshapeK^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Reshape_1
Т
Ygradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyIdentityHgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/ReshapeR^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Reshape*'
_output_shapes
:         
З
[gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependency_1IdentityJgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Reshape_1R^gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*
_output_shapes
: *
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/Reshape_1
─
Igradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ShapeShape;current_value_network/LayerNorm_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
ш
Hgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/SizeConst*
value	B :*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
с
Ggradients_1/current_value_network/LayerNorm_1/moments/variance_grad/addAddDcurrent_value_network/LayerNorm_1/moments/variance/reduction_indicesHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Size*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
щ
Ggradients_1/current_value_network/LayerNorm_1/moments/variance_grad/modFloorModGgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/addHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Size*
_output_shapes
:*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape
є
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape
я
Ogradients_1/current_value_network/LayerNorm_1/moments/variance_grad/range/startConst*
value	B : *\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
я
Ogradients_1/current_value_network/LayerNorm_1/moments/variance_grad/range/deltaConst*
value	B :*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
─
Igradients_1/current_value_network/LayerNorm_1/moments/variance_grad/rangeRangeOgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/range/startHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/SizeOgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/range/delta*

Tidx0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
ю
Ngradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Fill/valueConst*
_output_shapes
: *
value	B :*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0
Ё
Hgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/FillFillKgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_1Ngradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Fill/value*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:*
T0
а
Qgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/DynamicStitchDynamicStitchIgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/rangeGgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/modIgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ShapeHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Fill*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
N*#
_output_shapes
:         
э
Mgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
value	B :*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0
Д
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/MaximumMaximumQgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/DynamicStitchMgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum/y*#
_output_shapes
:         *
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape
є
Lgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/floordivFloorDivIgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ShapeKgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum*
_output_shapes
:*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape
╡
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ReshapeReshapeYgradients_1/current_value_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyQgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
╕
Hgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/TileTileKgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ReshapeLgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
╞
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2Shape;current_value_network/LayerNorm_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
╜
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_3Shape2current_value_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
є
Igradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ConstConst*
valueB: *^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
Ж
Hgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ProdProdKgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2Igradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Const*
	keep_dims( *

Tidx0*
T0*^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
ї
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Const_1Const*
valueB: *^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
К
Jgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Prod_1ProdKgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_3Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Const_1*
	keep_dims( *

Tidx0*
T0*^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
ё
Ogradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum_1/yConst*
value	B :*^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
Ў
Mgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum_1MaximumJgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Prod_1Ogradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum_1/y*
T0*^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
Ї
Ngradients_1/current_value_network/LayerNorm_1/moments/variance_grad/floordiv_1FloorDivHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/ProdMgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Maximum_1*
T0*^
_classT
RPloc:@gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
╨
Hgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/CastCastNgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Ь
Kgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/truedivRealDivHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/TileHgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
н
Rgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeShapecurrent_value_network/add_1*
_output_shapes
:*
T0*
out_type0
╩
Tgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1Shape6current_value_network/LayerNorm_1/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
т
bgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeTgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ц
Sgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/scalarConstL^gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
л
Pgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/mulMulSgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/scalarKgradients_1/current_value_network/LayerNorm_1/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
м
Pgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/subSubcurrent_value_network/add_16current_value_network/LayerNorm_1/moments/StopGradientL^gradients_1/current_value_network/LayerNorm_1/moments/variance_grad/truediv*'
_output_shapes
:         @*
T0
п
Rgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1MulPgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/mulPgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         @
╧
Pgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/SumSumRgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1bgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┼
Tgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeReshapePgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/SumRgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:         @*
T0
╙
Rgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1SumRgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1dgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╦
Vgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1ReshapeRgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1Tgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
с
Pgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/NegNegVgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
П
]gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_depsNoOpU^gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeQ^gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Neg
┬
egradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyIdentityTgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape^^gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*g
_class]
[Yloc:@gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape
╝
ggradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityPgradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Neg^^gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/Neg*'
_output_shapes
:         
а
Egradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ShapeShapecurrent_value_network/add_1*
_output_shapes
:*
T0*
out_type0
р
Dgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/SizeConst*
value	B :*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╤
Cgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/addAdd@current_value_network/LayerNorm_1/moments/mean/reduction_indicesDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Size*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
┘
Cgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/modFloorModCgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/addDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Size*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
ы
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_1Const*
valueB:*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
ч
Kgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/range/startConst*
value	B : *X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
ч
Kgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/range/deltaConst*
value	B :*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
░
Egradients_1/current_value_network/LayerNorm_1/moments/mean_grad/rangeRangeKgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/range/startDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/SizeKgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape
ц
Jgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Fill/valueConst*
value	B :*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
р
Dgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/FillFillGgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_1Jgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Fill/value*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
И
Mgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/DynamicStitchDynamicStitchEgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/rangeCgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/modEgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ShapeDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Fill*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
N*#
_output_shapes
:         
х
Igradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0
Ї
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/MaximumMaximumMgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/DynamicStitchIgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum/y*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*#
_output_shapes
:         
у
Hgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/floordivFloorDivEgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ShapeGgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
п
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ReshapeReshape[gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyMgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
м
Dgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/TileTileGgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ReshapeHgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
в
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2Shapecurrent_value_network/add_1*
T0*
out_type0*
_output_shapes
:
╡
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_3Shape.current_value_network/LayerNorm_1/moments/mean*
_output_shapes
:*
T0*
out_type0
ы
Egradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ConstConst*
valueB: *Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
Ў
Dgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ProdProdGgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2Egradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Const*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
э
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Const_1Const*
valueB: *Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
·
Fgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Prod_1ProdGgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_3Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Const_1*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
щ
Kgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum_1/yConst*
value	B :*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
ц
Igradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum_1MaximumFgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Prod_1Kgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum_1/y*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
T0
ф
Jgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/floordiv_1FloorDivDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/ProdIgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Maximum_1*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
╚
Dgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/CastCastJgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Р
Ggradients_1/current_value_network/LayerNorm_1/moments/mean_grad/truedivRealDivDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/TileDgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/Cast*'
_output_shapes
:         @*
T0
┴
gradients_1/AddN_1AddN[gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyegradients_1/current_value_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyGgradients_1/current_value_network/LayerNorm_1/moments/mean_grad/truediv*'
_output_shapes
:         @*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*
N
Р
2gradients_1/current_value_network/add_1_grad/ShapeShapecurrent_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:
~
4gradients_1/current_value_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
В
Bgradients_1/current_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/current_value_network/add_1_grad/Shape4gradients_1/current_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╧
0gradients_1/current_value_network/add_1_grad/SumSumgradients_1/AddN_1Bgradients_1/current_value_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
4gradients_1/current_value_network/add_1_grad/ReshapeReshape0gradients_1/current_value_network/add_1_grad/Sum2gradients_1/current_value_network/add_1_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
╙
2gradients_1/current_value_network/add_1_grad/Sum_1Sumgradients_1/AddN_1Dgradients_1/current_value_network/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
▐
6gradients_1/current_value_network/add_1_grad/Reshape_1Reshape2gradients_1/current_value_network/add_1_grad/Sum_14gradients_1/current_value_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
╡
=gradients_1/current_value_network/add_1_grad/tuple/group_depsNoOp5^gradients_1/current_value_network/add_1_grad/Reshape7^gradients_1/current_value_network/add_1_grad/Reshape_1
┬
Egradients_1/current_value_network/add_1_grad/tuple/control_dependencyIdentity4gradients_1/current_value_network/add_1_grad/Reshape>^gradients_1/current_value_network/add_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/current_value_network/add_1_grad/Reshape*'
_output_shapes
:         @
╗
Ggradients_1/current_value_network/add_1_grad/tuple/control_dependency_1Identity6gradients_1/current_value_network/add_1_grad/Reshape_1>^gradients_1/current_value_network/add_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/current_value_network/add_1_grad/Reshape_1*
_output_shapes
:@
Ч
6gradients_1/current_value_network/MatMul_1_grad/MatMulMatMulEgradients_1/current_value_network/add_1_grad/tuple/control_dependency6current_value_network/current_value_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
Ї
8gradients_1/current_value_network/MatMul_1_grad/MatMul_1MatMulcurrent_value_network/TanhEgradients_1/current_value_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
╝
@gradients_1/current_value_network/MatMul_1_grad/tuple/group_depsNoOp7^gradients_1/current_value_network/MatMul_1_grad/MatMul9^gradients_1/current_value_network/MatMul_1_grad/MatMul_1
╠
Hgradients_1/current_value_network/MatMul_1_grad/tuple/control_dependencyIdentity6gradients_1/current_value_network/MatMul_1_grad/MatMulA^gradients_1/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/current_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:         @
╔
Jgradients_1/current_value_network/MatMul_1_grad/tuple/control_dependency_1Identity8gradients_1/current_value_network/MatMul_1_grad/MatMul_1A^gradients_1/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/current_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
╪
4gradients_1/current_value_network/Tanh_grad/TanhGradTanhGradcurrent_value_network/TanhHgradients_1/current_value_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
╡
Fgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/ShapeShape/current_value_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
╡
Hgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape-current_value_network/LayerNorm/batchnorm/sub*
out_type0*
_output_shapes
:*
T0
╛
Vgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/ShapeHgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Щ
Dgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/SumSum4gradients_1/current_value_network/Tanh_grad/TanhGradVgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
Hgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeDgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/SumFgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:         @*
T0
Э
Fgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1Sum4gradients_1/current_value_network/Tanh_grad/TanhGradXgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
з
Jgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeFgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Sum_1Hgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:         @*
T0*
Tshape0
ё
Qgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpI^gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeK^gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
Т
Ygradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityHgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/ReshapeR^gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @
Ш
[gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityJgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1R^gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:         @
Я
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapecurrent_value_network/add*
out_type0*
_output_shapes
:*
T0
╡
Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape-current_value_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
╛
Vgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/ShapeHgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
З
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/mulMulYgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency-current_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
й
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/SumSumDgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/mulVgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeDgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/SumFgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
ї
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1Mulcurrent_value_network/addYgradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
п
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumFgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/mul_1Xgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
з
Jgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeFgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Sum_1Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ё
Qgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpI^gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeK^gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
Т
Ygradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityHgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/ReshapeR^gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:         @
Ш
[gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityJgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1R^gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
О
Dgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:
╡
Fgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape/current_value_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
╕
Tgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/ShapeFgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╝
Bgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/SumSum[gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Tgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
О
Fgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeBgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/SumDgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Shape*
_output_shapes
:@*
T0*
Tshape0
└
Dgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum[gradients_1/current_value_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Vgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
▓
Bgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/NegNegDgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
Я
Hgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeBgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/NegFgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:         @*
T0*
Tshape0
ы
Ogradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpG^gradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/ReshapeI^gradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1
¤
Wgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityFgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/ReshapeP^gradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:@
Р
Ygradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityHgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1P^gradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @
▓
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape,current_value_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
╡
Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape-current_value_network/LayerNorm/batchnorm/mul*
_output_shapes
:*
T0*
out_type0
╛
Vgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/ShapeHgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
З
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/mulMulYgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1-current_value_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
й
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/SumSumDgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/mulVgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeDgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/SumFgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
И
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul,current_value_network/LayerNorm/moments/meanYgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:         @*
T0
п
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumFgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/mul_1Xgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
з
Jgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeFgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Sum_1Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ё
Qgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpI^gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeK^gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
Т
Ygradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityHgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/ReshapeR^gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:         *
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape
Ш
[gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityJgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1R^gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:         @
ю
gradients_1/AddN_2AddN[gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1[gradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:         @
│
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/ShapeShape/current_value_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
Р
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
╕
Tgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/ShapeFgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╗
Bgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/mulMulgradients_1/AddN_2*current_value_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:         @
г
Bgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/SumSumBgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/mulTgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ы
Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeBgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/SumDgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
┬
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/mul_1Mul/current_value_network/LayerNorm/batchnorm/Rsqrtgradients_1/AddN_2*
T0*'
_output_shapes
:         @
й
Dgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Sum_1SumDgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/mul_1Vgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ф
Hgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeDgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Sum_1Fgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
ы
Ogradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpG^gradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/ReshapeI^gradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1
К
Wgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityFgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/ReshapeP^gradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*Y
_classO
MKloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Reshape
Г
Ygradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityHgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1P^gradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/Reshape_1*
_output_shapes
:@
У
Jgradients_1/current_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad/current_value_network/LayerNorm/batchnorm/RsqrtWgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
┤
Dgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/ShapeShape0current_value_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
Й
Fgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╕
Tgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/ShapeFgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
л
Bgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/SumSumJgradients_1/current_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradTgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ы
Fgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeBgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/SumDgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
п
Dgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Sum_1SumJgradients_1/current_value_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradVgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Р
Hgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeDgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Sum_1Fgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ы
Ogradients_1/current_value_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpG^gradients_1/current_value_network/LayerNorm/batchnorm/add_grad/ReshapeI^gradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Reshape_1
К
Wgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityFgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/ReshapeP^gradients_1/current_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:         
 
Ygradients_1/current_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityHgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Reshape_1P^gradients_1/current_value_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
└
Ggradients_1/current_value_network/LayerNorm/moments/variance_grad/ShapeShape9current_value_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
ф
Fgradients_1/current_value_network/LayerNorm/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape
┘
Egradients_1/current_value_network/LayerNorm/moments/variance_grad/addAddBcurrent_value_network/LayerNorm/moments/variance/reduction_indicesFgradients_1/current_value_network/LayerNorm/moments/variance_grad/Size*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
с
Egradients_1/current_value_network/LayerNorm/moments/variance_grad/modFloorModEgradients_1/current_value_network/LayerNorm/moments/variance_grad/addFgradients_1/current_value_network/LayerNorm/moments/variance_grad/Size*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
я
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape
ы
Mgradients_1/current_value_network/LayerNorm/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape
ы
Mgradients_1/current_value_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
║
Ggradients_1/current_value_network/LayerNorm/moments/variance_grad/rangeRangeMgradients_1/current_value_network/LayerNorm/moments/variance_grad/range/startFgradients_1/current_value_network/LayerNorm/moments/variance_grad/SizeMgradients_1/current_value_network/LayerNorm/moments/variance_grad/range/delta*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
ъ
Lgradients_1/current_value_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
ш
Fgradients_1/current_value_network/LayerNorm/moments/variance_grad/FillFillIgradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_1Lgradients_1/current_value_network/LayerNorm/moments/variance_grad/Fill/value*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
Ф
Ogradients_1/current_value_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchGgradients_1/current_value_network/LayerNorm/moments/variance_grad/rangeEgradients_1/current_value_network/LayerNorm/moments/variance_grad/modGgradients_1/current_value_network/LayerNorm/moments/variance_grad/ShapeFgradients_1/current_value_network/LayerNorm/moments/variance_grad/Fill*
N*#
_output_shapes
:         *
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape
щ
Kgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
№
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/MaximumMaximumOgradients_1/current_value_network/LayerNorm/moments/variance_grad/DynamicStitchKgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum/y*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*#
_output_shapes
:         
ы
Jgradients_1/current_value_network/LayerNorm/moments/variance_grad/floordivFloorDivGgradients_1/current_value_network/LayerNorm/moments/variance_grad/ShapeIgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum*
T0*Z
_classP
NLloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
п
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/ReshapeReshapeWgradients_1/current_value_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyOgradients_1/current_value_network/LayerNorm/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
▓
Fgradients_1/current_value_network/LayerNorm/moments/variance_grad/TileTileIgradients_1/current_value_network/LayerNorm/moments/variance_grad/ReshapeJgradients_1/current_value_network/LayerNorm/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
┬
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2Shape9current_value_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
╣
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_3Shape0current_value_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
я
Ggradients_1/current_value_network/LayerNorm/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2
■
Fgradients_1/current_value_network/LayerNorm/moments/variance_grad/ProdProdIgradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2Ggradients_1/current_value_network/LayerNorm/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2
ё
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2
В
Hgradients_1/current_value_network/LayerNorm/moments/variance_grad/Prod_1ProdIgradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_3Igradients_1/current_value_network/LayerNorm/moments/variance_grad/Const_1*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
э
Mgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
ю
Kgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum_1MaximumHgradients_1/current_value_network/LayerNorm/moments/variance_grad/Prod_1Mgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum_1/y*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: *
T0
ь
Lgradients_1/current_value_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivFgradients_1/current_value_network/LayerNorm/moments/variance_grad/ProdKgradients_1/current_value_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*\
_classR
PNloc:@gradients_1/current_value_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
╠
Fgradients_1/current_value_network/LayerNorm/moments/variance_grad/CastCastLgradients_1/current_value_network/LayerNorm/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Ц
Igradients_1/current_value_network/LayerNorm/moments/variance_grad/truedivRealDivFgradients_1/current_value_network/LayerNorm/moments/variance_grad/TileFgradients_1/current_value_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
й
Pgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapecurrent_value_network/add*
T0*
out_type0*
_output_shapes
:
╞
Rgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape4current_value_network/LayerNorm/moments/StopGradient*
out_type0*
_output_shapes
:*
T0
▄
`gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/ShapeRgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
т
Qgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/scalarConstJ^gradients_1/current_value_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
е
Ngradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/mulMulQgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/scalarIgradients_1/current_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
д
Ngradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/subSubcurrent_value_network/add4current_value_network/LayerNorm/moments/StopGradientJ^gradients_1/current_value_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
й
Pgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulNgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/mulNgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:         @*
T0
╔
Ngradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/SumSumPgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1`gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┐
Rgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeNgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/SumPgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
═
Pgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumPgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/mul_1bgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┼
Tgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapePgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Sum_1Rgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
▌
Ngradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/NegNegTgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:         *
T0
Й
[gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpS^gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/ReshapeO^gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
║
cgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentityRgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape\^gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:         @
┤
egradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityNgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Neg\^gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         *
T0*a
_classW
USloc:@gradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/Neg
Ь
Cgradients_1/current_value_network/LayerNorm/moments/mean_grad/ShapeShapecurrent_value_network/add*
T0*
out_type0*
_output_shapes
:
▄
Bgradients_1/current_value_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╔
Agradients_1/current_value_network/LayerNorm/moments/mean_grad/addAdd>current_value_network/LayerNorm/moments/mean/reduction_indicesBgradients_1/current_value_network/LayerNorm/moments/mean_grad/Size*
T0*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
╤
Agradients_1/current_value_network/LayerNorm/moments/mean_grad/modFloorModAgradients_1/current_value_network/LayerNorm/moments/mean_grad/addBgradients_1/current_value_network/LayerNorm/moments/mean_grad/Size*
T0*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
ч
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
у
Igradients_1/current_value_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
у
Igradients_1/current_value_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
ж
Cgradients_1/current_value_network/LayerNorm/moments/mean_grad/rangeRangeIgradients_1/current_value_network/LayerNorm/moments/mean_grad/range/startBgradients_1/current_value_network/LayerNorm/moments/mean_grad/SizeIgradients_1/current_value_network/LayerNorm/moments/mean_grad/range/delta*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
т
Hgradients_1/current_value_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╪
Bgradients_1/current_value_network/LayerNorm/moments/mean_grad/FillFillEgradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_1Hgradients_1/current_value_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape
№
Kgradients_1/current_value_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchCgradients_1/current_value_network/LayerNorm/moments/mean_grad/rangeAgradients_1/current_value_network/LayerNorm/moments/mean_grad/modCgradients_1/current_value_network/LayerNorm/moments/mean_grad/ShapeBgradients_1/current_value_network/LayerNorm/moments/mean_grad/Fill*
T0*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
N*#
_output_shapes
:         
с
Ggradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
ь
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/MaximumMaximumKgradients_1/current_value_network/LayerNorm/moments/mean_grad/DynamicStitchGgradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*#
_output_shapes
:         
█
Fgradients_1/current_value_network/LayerNorm/moments/mean_grad/floordivFloorDivCgradients_1/current_value_network/LayerNorm/moments/mean_grad/ShapeEgradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum*
T0*V
_classL
JHloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
й
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/ReshapeReshapeYgradients_1/current_value_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyKgradients_1/current_value_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
ж
Bgradients_1/current_value_network/LayerNorm/moments/mean_grad/TileTileEgradients_1/current_value_network/LayerNorm/moments/mean_grad/ReshapeFgradients_1/current_value_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
Ю
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2Shapecurrent_value_network/add*
T0*
out_type0*
_output_shapes
:
▒
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_3Shape,current_value_network/LayerNorm/moments/mean*
out_type0*
_output_shapes
:*
T0
ч
Cgradients_1/current_value_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
ю
Bgradients_1/current_value_network/LayerNorm/moments/mean_grad/ProdProdEgradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2Cgradients_1/current_value_network/LayerNorm/moments/mean_grad/Const*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
щ
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2
Є
Dgradients_1/current_value_network/LayerNorm/moments/mean_grad/Prod_1ProdEgradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_3Egradients_1/current_value_network/LayerNorm/moments/mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
х
Igradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
▐
Ggradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum_1MaximumDgradients_1/current_value_network/LayerNorm/moments/mean_grad/Prod_1Igradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2
▄
Hgradients_1/current_value_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivBgradients_1/current_value_network/LayerNorm/moments/mean_grad/ProdGgradients_1/current_value_network/LayerNorm/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*X
_classN
LJloc:@gradients_1/current_value_network/LayerNorm/moments/mean_grad/Shape_2
─
Bgradients_1/current_value_network/LayerNorm/moments/mean_grad/CastCastHgradients_1/current_value_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
К
Egradients_1/current_value_network/LayerNorm/moments/mean_grad/truedivRealDivBgradients_1/current_value_network/LayerNorm/moments/mean_grad/TileBgradients_1/current_value_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:         @
╣
gradients_1/AddN_3AddNYgradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencycgradients_1/current_value_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyEgradients_1/current_value_network/LayerNorm/moments/mean_grad/truediv*
N*'
_output_shapes
:         @*
T0*[
_classQ
OMloc:@gradients_1/current_value_network/LayerNorm/batchnorm/mul_1_grad/Reshape
М
0gradients_1/current_value_network/add_grad/ShapeShapecurrent_value_network/MatMul*
T0*
out_type0*
_output_shapes
:
|
2gradients_1/current_value_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
№
@gradients_1/current_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_1/current_value_network/add_grad/Shape2gradients_1/current_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╦
.gradients_1/current_value_network/add_grad/SumSumgradients_1/AddN_3@gradients_1/current_value_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
▀
2gradients_1/current_value_network/add_grad/ReshapeReshape.gradients_1/current_value_network/add_grad/Sum0gradients_1/current_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
╧
0gradients_1/current_value_network/add_grad/Sum_1Sumgradients_1/AddN_3Bgradients_1/current_value_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╪
4gradients_1/current_value_network/add_grad/Reshape_1Reshape0gradients_1/current_value_network/add_grad/Sum_12gradients_1/current_value_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
п
;gradients_1/current_value_network/add_grad/tuple/group_depsNoOp3^gradients_1/current_value_network/add_grad/Reshape5^gradients_1/current_value_network/add_grad/Reshape_1
║
Cgradients_1/current_value_network/add_grad/tuple/control_dependencyIdentity2gradients_1/current_value_network/add_grad/Reshape<^gradients_1/current_value_network/add_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/current_value_network/add_grad/Reshape*'
_output_shapes
:         @*
T0
│
Egradients_1/current_value_network/add_grad/tuple/control_dependency_1Identity4gradients_1/current_value_network/add_grad/Reshape_1<^gradients_1/current_value_network/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/current_value_network/add_grad/Reshape_1*
_output_shapes
:@
У
4gradients_1/current_value_network/MatMul_grad/MatMulMatMulCgradients_1/current_value_network/add_grad/tuple/control_dependency6current_value_network/current_value_network/fc0/w/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
ф
6gradients_1/current_value_network/MatMul_grad/MatMul_1MatMulobservations_1Cgradients_1/current_value_network/add_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
╢
>gradients_1/current_value_network/MatMul_grad/tuple/group_depsNoOp5^gradients_1/current_value_network/MatMul_grad/MatMul7^gradients_1/current_value_network/MatMul_grad/MatMul_1
─
Fgradients_1/current_value_network/MatMul_grad/tuple/control_dependencyIdentity4gradients_1/current_value_network/MatMul_grad/MatMul?^gradients_1/current_value_network/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/current_value_network/MatMul_grad/MatMul*'
_output_shapes
:         
┴
Hgradients_1/current_value_network/MatMul_grad/tuple/control_dependency_1Identity6gradients_1/current_value_network/MatMul_grad/MatMul_1?^gradients_1/current_value_network/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*I
_class?
=;loc:@gradients_1/current_value_network/MatMul_grad/MatMul_1
Щ
beta1_power_1/initial_valueConst*
valueB
 *fff?*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
к
beta1_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *7
_class-
+)loc:@current_value_network/LayerNorm/beta*
	container 
═
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
З
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta
Щ
beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *w╛?*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
dtype0
к
beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *7
_class-
+)loc:@current_value_network/LayerNorm/beta*
	container *
shape: 
═
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
З
beta2_power_1/readIdentitybeta2_power_1*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
_output_shapes
: 
у
Hcurrent_value_network/current_value_network/fc0/w/Adam/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ё
6current_value_network/current_value_network/fc0/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
	container *
shape
:@
с
=current_value_network/current_value_network/fc0/w/Adam/AssignAssign6current_value_network/current_value_network/fc0/w/AdamHcurrent_value_network/current_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
ю
;current_value_network/current_value_network/fc0/w/Adam/readIdentity6current_value_network/current_value_network/fc0/w/Adam*
_output_shapes

:@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w
х
Jcurrent_value_network/current_value_network/fc0/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
Є
8current_value_network/current_value_network/fc0/w/Adam_1
VariableV2*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
ч
?current_value_network/current_value_network/fc0/w/Adam_1/AssignAssign8current_value_network/current_value_network/fc0/w/Adam_1Jcurrent_value_network/current_value_network/fc0/w/Adam_1/Initializer/zeros*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
Є
=current_value_network/current_value_network/fc0/w/Adam_1/readIdentity8current_value_network/current_value_network/fc0/w/Adam_1*
_output_shapes

:@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w
█
Hcurrent_value_network/current_value_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
valueB@*    
ш
6current_value_network/current_value_network/fc0/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
	container *
shape:@
▌
=current_value_network/current_value_network/fc0/b/Adam/AssignAssign6current_value_network/current_value_network/fc0/b/AdamHcurrent_value_network/current_value_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
ъ
;current_value_network/current_value_network/fc0/b/Adam/readIdentity6current_value_network/current_value_network/fc0/b/Adam*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
_output_shapes
:@
▌
Jcurrent_value_network/current_value_network/fc0/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ъ
8current_value_network/current_value_network/fc0/b/Adam_1
VariableV2*
_output_shapes
:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
	container *
shape:@*
dtype0
у
?current_value_network/current_value_network/fc0/b/Adam_1/AssignAssign8current_value_network/current_value_network/fc0/b/Adam_1Jcurrent_value_network/current_value_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
ю
=current_value_network/current_value_network/fc0/b/Adam_1/readIdentity8current_value_network/current_value_network/fc0/b/Adam_1*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
_output_shapes
:@*
T0
┴
;current_value_network/LayerNorm/beta/Adam/Initializer/zerosConst*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╬
)current_value_network/LayerNorm/beta/Adam
VariableV2*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@current_value_network/LayerNorm/beta*
	container *
shape:@*
dtype0
й
0current_value_network/LayerNorm/beta/Adam/AssignAssign)current_value_network/LayerNorm/beta/Adam;current_value_network/LayerNorm/beta/Adam/Initializer/zeros*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
├
.current_value_network/LayerNorm/beta/Adam/readIdentity)current_value_network/LayerNorm/beta/Adam*
_output_shapes
:@*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta
├
=current_value_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
valueB@*    
╨
+current_value_network/LayerNorm/beta/Adam_1
VariableV2*
shared_name *7
_class-
+)loc:@current_value_network/LayerNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
п
2current_value_network/LayerNorm/beta/Adam_1/AssignAssign+current_value_network/LayerNorm/beta/Adam_1=current_value_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
╟
0current_value_network/LayerNorm/beta/Adam_1/readIdentity+current_value_network/LayerNorm/beta/Adam_1*
_output_shapes
:@*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta
├
<current_value_network/LayerNorm/gamma/Adam/Initializer/zerosConst*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╨
*current_value_network/LayerNorm/gamma/Adam
VariableV2*
shared_name *8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
н
1current_value_network/LayerNorm/gamma/Adam/AssignAssign*current_value_network/LayerNorm/gamma/Adam<current_value_network/LayerNorm/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*8
_class.
,*loc:@current_value_network/LayerNorm/gamma
╞
/current_value_network/LayerNorm/gamma/Adam/readIdentity*current_value_network/LayerNorm/gamma/Adam*
_output_shapes
:@*
T0*8
_class.
,*loc:@current_value_network/LayerNorm/gamma
┼
>current_value_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
valueB@*    
╥
,current_value_network/LayerNorm/gamma/Adam_1
VariableV2*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
	container *
shape:@*
dtype0
│
3current_value_network/LayerNorm/gamma/Adam_1/AssignAssign,current_value_network/LayerNorm/gamma/Adam_1>current_value_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
T0*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
╩
1current_value_network/LayerNorm/gamma/Adam_1/readIdentity,current_value_network/LayerNorm/gamma/Adam_1*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
_output_shapes
:@*
T0
у
Hcurrent_value_network/current_value_network/fc1/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@@*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
valueB@@*    
Ё
6current_value_network/current_value_network/fc1/w/Adam
VariableV2*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
с
=current_value_network/current_value_network/fc1/w/Adam/AssignAssign6current_value_network/current_value_network/fc1/w/AdamHcurrent_value_network/current_value_network/fc1/w/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ю
;current_value_network/current_value_network/fc1/w/Adam/readIdentity6current_value_network/current_value_network/fc1/w/Adam*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
х
Jcurrent_value_network/current_value_network/fc1/w/Adam_1/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
Є
8current_value_network/current_value_network/fc1/w/Adam_1
VariableV2*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
ч
?current_value_network/current_value_network/fc1/w/Adam_1/AssignAssign8current_value_network/current_value_network/fc1/w/Adam_1Jcurrent_value_network/current_value_network/fc1/w/Adam_1/Initializer/zeros*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
Є
=current_value_network/current_value_network/fc1/w/Adam_1/readIdentity8current_value_network/current_value_network/fc1/w/Adam_1*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
█
Hcurrent_value_network/current_value_network/fc1/b/Adam/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ш
6current_value_network/current_value_network/fc1/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
	container *
shape:@
▌
=current_value_network/current_value_network/fc1/b/Adam/AssignAssign6current_value_network/current_value_network/fc1/b/AdamHcurrent_value_network/current_value_network/fc1/b/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
validate_shape(
ъ
;current_value_network/current_value_network/fc1/b/Adam/readIdentity6current_value_network/current_value_network/fc1/b/Adam*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
_output_shapes
:@
▌
Jcurrent_value_network/current_value_network/fc1/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ъ
8current_value_network/current_value_network/fc1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
	container *
shape:@
у
?current_value_network/current_value_network/fc1/b/Adam_1/AssignAssign8current_value_network/current_value_network/fc1/b/Adam_1Jcurrent_value_network/current_value_network/fc1/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b
ю
=current_value_network/current_value_network/fc1/b/Adam_1/readIdentity8current_value_network/current_value_network/fc1/b/Adam_1*
_output_shapes
:@*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b
┼
=current_value_network/LayerNorm_1/beta/Adam/Initializer/zerosConst*
_output_shapes
:@*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
valueB@*    *
dtype0
╥
+current_value_network/LayerNorm_1/beta/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
	container *
shape:@
▒
2current_value_network/LayerNorm_1/beta/Adam/AssignAssign+current_value_network/LayerNorm_1/beta/Adam=current_value_network/LayerNorm_1/beta/Adam/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
╔
0current_value_network/LayerNorm_1/beta/Adam/readIdentity+current_value_network/LayerNorm_1/beta/Adam*
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
_output_shapes
:@
╟
?current_value_network/LayerNorm_1/beta/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╘
-current_value_network/LayerNorm_1/beta/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_value_network/LayerNorm_1/beta
╖
4current_value_network/LayerNorm_1/beta/Adam_1/AssignAssign-current_value_network/LayerNorm_1/beta/Adam_1?current_value_network/LayerNorm_1/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
═
2current_value_network/LayerNorm_1/beta/Adam_1/readIdentity-current_value_network/LayerNorm_1/beta/Adam_1*
_output_shapes
:@*
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta
╟
>current_value_network/LayerNorm_1/gamma/Adam/Initializer/zerosConst*
_output_shapes
:@*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
valueB@*    *
dtype0
╘
,current_value_network/LayerNorm_1/gamma/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
	container 
╡
3current_value_network/LayerNorm_1/gamma/Adam/AssignAssign,current_value_network/LayerNorm_1/gamma/Adam>current_value_network/LayerNorm_1/gamma/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╠
1current_value_network/LayerNorm_1/gamma/Adam/readIdentity,current_value_network/LayerNorm_1/gamma/Adam*
_output_shapes
:@*
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma
╔
@current_value_network/LayerNorm_1/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
valueB@*    
╓
.current_value_network/LayerNorm_1/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
	container *
shape:@
╗
5current_value_network/LayerNorm_1/gamma/Adam_1/AssignAssign.current_value_network/LayerNorm_1/gamma/Adam_1@current_value_network/LayerNorm_1/gamma/Adam_1/Initializer/zeros*
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
╨
3current_value_network/LayerNorm_1/gamma/Adam_1/readIdentity.current_value_network/LayerNorm_1/gamma/Adam_1*
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
_output_shapes
:@
у
Hcurrent_value_network/current_value_network/out/w/Adam/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ё
6current_value_network/current_value_network/out/w/Adam
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/out/w
с
=current_value_network/current_value_network/out/w/Adam/AssignAssign6current_value_network/current_value_network/out/w/AdamHcurrent_value_network/current_value_network/out/w/Adam/Initializer/zeros*D
_class:
86loc:@current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
ю
;current_value_network/current_value_network/out/w/Adam/readIdentity6current_value_network/current_value_network/out/w/Adam*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w*
_output_shapes

:@
х
Jcurrent_value_network/current_value_network/out/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@*D
_class:
86loc:@current_value_network/current_value_network/out/w*
valueB@*    *
dtype0
Є
8current_value_network/current_value_network/out/w/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/out/w*
	container 
ч
?current_value_network/current_value_network/out/w/Adam_1/AssignAssign8current_value_network/current_value_network/out/w/Adam_1Jcurrent_value_network/current_value_network/out/w/Adam_1/Initializer/zeros*D
_class:
86loc:@current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
Є
=current_value_network/current_value_network/out/w/Adam_1/readIdentity8current_value_network/current_value_network/out/w/Adam_1*D
_class:
86loc:@current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
█
Hcurrent_value_network/current_value_network/out/b/Adam/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ш
6current_value_network/current_value_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@current_value_network/current_value_network/out/b*
	container *
shape:
▌
=current_value_network/current_value_network/out/b/Adam/AssignAssign6current_value_network/current_value_network/out/b/AdamHcurrent_value_network/current_value_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b*
validate_shape(
ъ
;current_value_network/current_value_network/out/b/Adam/readIdentity6current_value_network/current_value_network/out/b/Adam*
_output_shapes
:*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b
▌
Jcurrent_value_network/current_value_network/out/b/Adam_1/Initializer/zerosConst*D
_class:
86loc:@current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ъ
8current_value_network/current_value_network/out/b/Adam_1
VariableV2*D
_class:
86loc:@current_value_network/current_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
у
?current_value_network/current_value_network/out/b/Adam_1/AssignAssign8current_value_network/current_value_network/out/b/Adam_1Jcurrent_value_network/current_value_network/out/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b
ю
=current_value_network/current_value_network/out/b/Adam_1/readIdentity8current_value_network/current_value_network/out/b/Adam_1*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b*
_output_shapes
:
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
└
IAdam_1/update_current_value_network/current_value_network/fc0/w/ApplyAdam	ApplyAdam1current_value_network/current_value_network/fc0/w6current_value_network/current_value_network/fc0/w/Adam8current_value_network/current_value_network/fc0/w/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/current_value_network/MatMul_grad/tuple/control_dependency_1*D
_class:
86loc:@current_value_network/current_value_network/fc0/w*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0
╣
IAdam_1/update_current_value_network/current_value_network/fc0/b/ApplyAdam	ApplyAdam1current_value_network/current_value_network/fc0/b6current_value_network/current_value_network/fc0/b/Adam8current_value_network/current_value_network/fc0/b/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonEgradients_1/current_value_network/add_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
use_nesterov( *
_output_shapes
:@*
use_locking( 
К
<Adam_1/update_current_value_network/LayerNorm/beta/ApplyAdam	ApplyAdam$current_value_network/LayerNorm/beta)current_value_network/LayerNorm/beta/Adam+current_value_network/LayerNorm/beta/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/current_value_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes
:@*
use_locking( *
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
use_nesterov( 
С
=Adam_1/update_current_value_network/LayerNorm/gamma/ApplyAdam	ApplyAdam%current_value_network/LayerNorm/gamma*current_value_network/LayerNorm/gamma/Adam,current_value_network/LayerNorm/gamma/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/current_value_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:@
┬
IAdam_1/update_current_value_network/current_value_network/fc1/w/ApplyAdam	ApplyAdam1current_value_network/current_value_network/fc1/w6current_value_network/current_value_network/fc1/w/Adam8current_value_network/current_value_network/fc1/w/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonJgradients_1/current_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
use_nesterov( *
_output_shapes

:@@
╗
IAdam_1/update_current_value_network/current_value_network/fc1/b/ApplyAdam	ApplyAdam1current_value_network/current_value_network/fc1/b6current_value_network/current_value_network/fc1/b/Adam8current_value_network/current_value_network/fc1/b/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/current_value_network/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
use_nesterov( *
_output_shapes
:@
Ц
>Adam_1/update_current_value_network/LayerNorm_1/beta/ApplyAdam	ApplyAdam&current_value_network/LayerNorm_1/beta+current_value_network/LayerNorm_1/beta/Adam-current_value_network/LayerNorm_1/beta/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/current_value_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta
Э
?Adam_1/update_current_value_network/LayerNorm_1/gamma/ApplyAdam	ApplyAdam'current_value_network/LayerNorm_1/gamma,current_value_network/LayerNorm_1/gamma/Adam.current_value_network/LayerNorm_1/gamma/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon[gradients_1/current_value_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma
┬
IAdam_1/update_current_value_network/current_value_network/out/w/ApplyAdam	ApplyAdam1current_value_network/current_value_network/out/w6current_value_network/current_value_network/out/w/Adam8current_value_network/current_value_network/out/w/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonJgradients_1/current_value_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w*
use_nesterov( *
_output_shapes

:@
╗
IAdam_1/update_current_value_network/current_value_network/out/b/ApplyAdam	ApplyAdam1current_value_network/current_value_network/out/b6current_value_network/current_value_network/out/b/Adam8current_value_network/current_value_network/out/b/Adam_1beta1_power_1/readbeta2_power_1/readlearning_rate_1Adam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/current_value_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b
╫

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1J^Adam_1/update_current_value_network/current_value_network/fc0/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc0/b/ApplyAdam=^Adam_1/update_current_value_network/LayerNorm/beta/ApplyAdam>^Adam_1/update_current_value_network/LayerNorm/gamma/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc1/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc1/b/ApplyAdam?^Adam_1/update_current_value_network/LayerNorm_1/beta/ApplyAdam@^Adam_1/update_current_value_network/LayerNorm_1/gamma/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/out/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/out/b/ApplyAdam*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
_output_shapes
: 
╡
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(
┘
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2J^Adam_1/update_current_value_network/current_value_network/fc0/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc0/b/ApplyAdam=^Adam_1/update_current_value_network/LayerNorm/beta/ApplyAdam>^Adam_1/update_current_value_network/LayerNorm/gamma/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc1/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc1/b/ApplyAdam?^Adam_1/update_current_value_network/LayerNorm_1/beta/ApplyAdam@^Adam_1/update_current_value_network/LayerNorm_1/gamma/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/out/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/out/b/ApplyAdam*
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
_output_shapes
: 
╣
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
·
Adam_1NoOpJ^Adam_1/update_current_value_network/current_value_network/fc0/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc0/b/ApplyAdam=^Adam_1/update_current_value_network/LayerNorm/beta/ApplyAdam>^Adam_1/update_current_value_network/LayerNorm/gamma/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc1/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/fc1/b/ApplyAdam?^Adam_1/update_current_value_network/LayerNorm_1/beta/ApplyAdam@^Adam_1/update_current_value_network/LayerNorm_1/gamma/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/out/w/ApplyAdamJ^Adam_1/update_current_value_network/current_value_network/out/b/ApplyAdam^Adam_1/Assign^Adam_1/Assign_1
щ
	Assign_40Assign#target_value_network/LayerNorm/beta)current_value_network/LayerNorm/beta/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*6
_class,
*(loc:@target_value_network/LayerNorm/beta
ь
	Assign_41Assign$target_value_network/LayerNorm/gamma*current_value_network/LayerNorm/gamma/read*
_output_shapes
:@*
use_locking( *
T0*7
_class-
+)loc:@target_value_network/LayerNorm/gamma*
validate_shape(
я
	Assign_42Assign%target_value_network/LayerNorm_1/beta+current_value_network/LayerNorm_1/beta/read*
use_locking( *
T0*8
_class.
,*loc:@target_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
Є
	Assign_43Assign&target_value_network/LayerNorm_1/gamma,current_value_network/LayerNorm_1/gamma/read*
use_locking( *
T0*9
_class/
-+loc:@target_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
О
	Assign_44Assign/target_value_network/target_value_network/fc0/b6current_value_network/current_value_network/fc0/b/read*B
_class8
64loc:@target_value_network/target_value_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
Т
	Assign_45Assign/target_value_network/target_value_network/fc0/w6current_value_network/current_value_network/fc0/w/read*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
validate_shape(*
_output_shapes

:@
О
	Assign_46Assign/target_value_network/target_value_network/fc1/b6current_value_network/current_value_network/fc1/b/read*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
Т
	Assign_47Assign/target_value_network/target_value_network/fc1/w6current_value_network/current_value_network/fc1/w/read*
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 
О
	Assign_48Assign/target_value_network/target_value_network/out/b6current_value_network/current_value_network/out/b/read*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/out/b
Т
	Assign_49Assign/target_value_network/target_value_network/out/w6current_value_network/current_value_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w*
validate_shape(
М
group_deps_3NoOp
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
ф
	Assign_50Assign!best_value_network/LayerNorm/beta(target_value_network/LayerNorm/beta/read*
use_locking( *
T0*4
_class*
(&loc:@best_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
ч
	Assign_51Assign"best_value_network/LayerNorm/gamma)target_value_network/LayerNorm/gamma/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*5
_class+
)'loc:@best_value_network/LayerNorm/gamma
ъ
	Assign_52Assign#best_value_network/LayerNorm_1/beta*target_value_network/LayerNorm_1/beta/read*6
_class,
*(loc:@best_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
э
	Assign_53Assign$best_value_network/LayerNorm_1/gamma+target_value_network/LayerNorm_1/gamma/read*
T0*7
_class-
+)loc:@best_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( 
Д
	Assign_54Assign+best_value_network/best_value_network/fc0/b4target_value_network/target_value_network/fc0/b/read*
use_locking( *
T0*>
_class4
20loc:@best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
И
	Assign_55Assign+best_value_network/best_value_network/fc0/w4target_value_network/target_value_network/fc0/w/read*>
_class4
20loc:@best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
Д
	Assign_56Assign+best_value_network/best_value_network/fc1/b4target_value_network/target_value_network/fc1/b/read*
use_locking( *
T0*>
_class4
20loc:@best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
И
	Assign_57Assign+best_value_network/best_value_network/fc1/w4target_value_network/target_value_network/fc1/w/read*>
_class4
20loc:@best_value_network/best_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0
Д
	Assign_58Assign+best_value_network/best_value_network/out/b4target_value_network/target_value_network/out/b/read*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*>
_class4
20loc:@best_value_network/best_value_network/out/b
И
	Assign_59Assign+best_value_network/best_value_network/out/w4target_value_network/target_value_network/out/w/read*
use_locking( *
T0*>
_class4
20loc:@best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@
М
group_deps_4NoOp
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
ц
	Assign_60Assign#target_value_network/LayerNorm/beta&best_value_network/LayerNorm/beta/read*
_output_shapes
:@*
use_locking( *
T0*6
_class,
*(loc:@target_value_network/LayerNorm/beta*
validate_shape(
щ
	Assign_61Assign$target_value_network/LayerNorm/gamma'best_value_network/LayerNorm/gamma/read*
use_locking( *
T0*7
_class-
+)loc:@target_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
ь
	Assign_62Assign%target_value_network/LayerNorm_1/beta(best_value_network/LayerNorm_1/beta/read*
use_locking( *
T0*8
_class.
,*loc:@target_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
я
	Assign_63Assign&target_value_network/LayerNorm_1/gamma)best_value_network/LayerNorm_1/gamma/read*
use_locking( *
T0*9
_class/
-+loc:@target_value_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
И
	Assign_64Assign/target_value_network/target_value_network/fc0/b0best_value_network/best_value_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/fc0/b*
validate_shape(
М
	Assign_65Assign/target_value_network/target_value_network/fc0/w0best_value_network/best_value_network/fc0/w/read*B
_class8
64loc:@target_value_network/target_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
И
	Assign_66Assign/target_value_network/target_value_network/fc1/b0best_value_network/best_value_network/fc1/b/read*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
М
	Assign_67Assign/target_value_network/target_value_network/fc1/w0best_value_network/best_value_network/fc1/w/read*
_output_shapes

:@@*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/fc1/w*
validate_shape(
И
	Assign_68Assign/target_value_network/target_value_network/out/b0best_value_network/best_value_network/out/b/read*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/out/b*
validate_shape(*
_output_shapes
:
М
	Assign_69Assign/target_value_network/target_value_network/out/w0best_value_network/best_value_network/out/w/read*
use_locking( *
T0*B
_class8
64loc:@target_value_network/target_value_network/out/w*
validate_shape(*
_output_shapes

:@
ш
	Assign_70Assign$current_value_network/LayerNorm/beta&best_value_network/LayerNorm/beta/read*
use_locking( *
T0*7
_class-
+)loc:@current_value_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
ы
	Assign_71Assign%current_value_network/LayerNorm/gamma'best_value_network/LayerNorm/gamma/read*8
_class.
,*loc:@current_value_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
ю
	Assign_72Assign&current_value_network/LayerNorm_1/beta(best_value_network/LayerNorm_1/beta/read*9
_class/
-+loc:@current_value_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
ё
	Assign_73Assign'current_value_network/LayerNorm_1/gamma)best_value_network/LayerNorm_1/gamma/read*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@current_value_network/LayerNorm_1/gamma*
validate_shape(
М
	Assign_74Assign1current_value_network/current_value_network/fc0/b0best_value_network/best_value_network/fc0/b/read*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
Р
	Assign_75Assign1current_value_network/current_value_network/fc0/w0best_value_network/best_value_network/fc0/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/fc0/w
М
	Assign_76Assign1current_value_network/current_value_network/fc1/b0best_value_network/best_value_network/fc1/b/read*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
Р
	Assign_77Assign1current_value_network/current_value_network/fc1/w0best_value_network/best_value_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
М
	Assign_78Assign1current_value_network/current_value_network/out/b0best_value_network/best_value_network/out/b/read*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
Р
	Assign_79Assign1current_value_network/current_value_network/out/w0best_value_network/best_value_network/out/w/read*
T0*D
_class:
86loc:@current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( 
Д
group_deps_5NoOp
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71
^Assign_72
^Assign_73
^Assign_74
^Assign_75
^Assign_76
^Assign_77
^Assign_78
^Assign_79
T
learning_rate_2Placeholder*
dtype0*
_output_shapes
:*
shape:
L
std_devPlaceholder*
dtype0*
_output_shapes
:*
shape:
q
observations_2Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
l
	actions_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
m

advantagesPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
a
Mean_2Mean
advantagesConst_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
э
Tcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
▀
Rcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
▀
Rcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
█
\current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformTcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
seed2╧
ъ
Rcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/subSubRcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxRcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w
№
Rcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulMul\current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
ю
Ncurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniformAddRcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulRcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
я
3current_policy_network/current_policy_network/fc0/w
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w
у
:current_policy_network/current_policy_network/fc0/w/AssignAssign3current_policy_network/current_policy_network/fc0/wNcurrent_policy_network/current_policy_network/fc0/w/Initializer/random_uniform*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
ъ
8current_policy_network/current_policy_network/fc0/w/readIdentity3current_policy_network/current_policy_network/fc0/w*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
┌
Ecurrent_policy_network/current_policy_network/fc0/b/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ч
3current_policy_network/current_policy_network/fc0/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
	container 
╓
:current_policy_network/current_policy_network/fc0/b/AssignAssign3current_policy_network/current_policy_network/fc0/bEcurrent_policy_network/current_policy_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b
ц
8current_policy_network/current_policy_network/fc0/b/readIdentity3current_policy_network/current_policy_network/fc0/b*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
╔
current_policy_network/MatMulMatMulobservations_28current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
м
current_policy_network/addAddcurrent_policy_network/MatMul8current_policy_network/current_policy_network/fc0/b/read*'
_output_shapes
:         @*
T0
╛
7current_policy_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:@*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
valueB@*    *
dtype0
╦
%current_policy_network/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape:@
Ю
,current_policy_network/LayerNorm/beta/AssignAssign%current_policy_network/LayerNorm/beta7current_policy_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
╝
*current_policy_network/LayerNorm/beta/readIdentity%current_policy_network/LayerNorm/beta*
_output_shapes
:@*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta
┐
7current_policy_network/LayerNorm/gamma/Initializer/onesConst*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
═
&current_policy_network/LayerNorm/gamma
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
	container 
б
-current_policy_network/LayerNorm/gamma/AssignAssign&current_policy_network/LayerNorm/gamma7current_policy_network/LayerNorm/gamma/Initializer/ones*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
┐
+current_policy_network/LayerNorm/gamma/readIdentity&current_policy_network/LayerNorm/gamma*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
_output_shapes
:@
Й
?current_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
с
-current_policy_network/LayerNorm/moments/meanMeancurrent_policy_network/add?current_policy_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
ж
5current_policy_network/LayerNorm/moments/StopGradientStopGradient-current_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:         
╘
:current_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencecurrent_policy_network/add5current_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:         @
Н
Ccurrent_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Й
1current_policy_network/LayerNorm/moments/varianceMean:current_policy_network/LayerNorm/moments/SquaredDifferenceCcurrent_policy_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
u
0current_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╠
.current_policy_network/LayerNorm/batchnorm/addAdd1current_policy_network/LayerNorm/moments/variance0current_policy_network/LayerNorm/batchnorm/add/y*'
_output_shapes
:         *
T0
Ы
0current_policy_network/LayerNorm/batchnorm/RsqrtRsqrt.current_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
╞
.current_policy_network/LayerNorm/batchnorm/mulMul0current_policy_network/LayerNorm/batchnorm/Rsqrt+current_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:         @
╡
0current_policy_network/LayerNorm/batchnorm/mul_1Mulcurrent_policy_network/add.current_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
╚
0current_policy_network/LayerNorm/batchnorm/mul_2Mul-current_policy_network/LayerNorm/moments/mean.current_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┼
.current_policy_network/LayerNorm/batchnorm/subSub*current_policy_network/LayerNorm/beta/read0current_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╦
0current_policy_network/LayerNorm/batchnorm/add_1Add0current_policy_network/LayerNorm/batchnorm/mul_1.current_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:         @
З
current_policy_network/TanhTanh0current_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:         @*
T0
э
Tcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB"@   @   *
dtype0
▀
Rcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/minConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
▀
Rcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
█
\current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformTcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shape*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
seed2Ў*
dtype0*
_output_shapes

:@@*

seed
ъ
Rcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/subSubRcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxRcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
№
Rcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulMul\current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
ю
Ncurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniformAddRcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulRcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
я
3current_policy_network/current_policy_network/fc1/w
VariableV2*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
у
:current_policy_network/current_policy_network/fc1/w/AssignAssign3current_policy_network/current_policy_network/fc1/wNcurrent_policy_network/current_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ъ
8current_policy_network/current_policy_network/fc1/w/readIdentity3current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
┌
Ecurrent_policy_network/current_policy_network/fc1/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
valueB@*    
ч
3current_policy_network/current_policy_network/fc1/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
	container 
╓
:current_policy_network/current_policy_network/fc1/b/AssignAssign3current_policy_network/current_policy_network/fc1/bEcurrent_policy_network/current_policy_network/fc1/b/Initializer/zeros*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
ц
8current_policy_network/current_policy_network/fc1/b/readIdentity3current_policy_network/current_policy_network/fc1/b*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
╪
current_policy_network/MatMul_1MatMulcurrent_policy_network/Tanh8current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
░
current_policy_network/add_1Addcurrent_policy_network/MatMul_18current_policy_network/current_policy_network/fc1/b/read*
T0*'
_output_shapes
:         @
┬
9current_policy_network/LayerNorm_1/beta/Initializer/zerosConst*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╧
'current_policy_network/LayerNorm_1/beta
VariableV2*
shared_name *:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
ж
.current_policy_network/LayerNorm_1/beta/AssignAssign'current_policy_network/LayerNorm_1/beta9current_policy_network/LayerNorm_1/beta/Initializer/zeros*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
┬
,current_policy_network/LayerNorm_1/beta/readIdentity'current_policy_network/LayerNorm_1/beta*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
_output_shapes
:@*
T0
├
9current_policy_network/LayerNorm_1/gamma/Initializer/onesConst*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╤
(current_policy_network/LayerNorm_1/gamma
VariableV2*
shared_name *;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
й
/current_policy_network/LayerNorm_1/gamma/AssignAssign(current_policy_network/LayerNorm_1/gamma9current_policy_network/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
┼
-current_policy_network/LayerNorm_1/gamma/readIdentity(current_policy_network/LayerNorm_1/gamma*
_output_shapes
:@*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma
Л
Acurrent_policy_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ч
/current_policy_network/LayerNorm_1/moments/meanMeancurrent_policy_network/add_1Acurrent_policy_network/LayerNorm_1/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
к
7current_policy_network/LayerNorm_1/moments/StopGradientStopGradient/current_policy_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
┌
<current_policy_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencecurrent_policy_network/add_17current_policy_network/LayerNorm_1/moments/StopGradient*
T0*'
_output_shapes
:         @
П
Ecurrent_policy_network/LayerNorm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
П
3current_policy_network/LayerNorm_1/moments/varianceMean<current_policy_network/LayerNorm_1/moments/SquaredDifferenceEcurrent_policy_network/LayerNorm_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
w
2current_policy_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╥
0current_policy_network/LayerNorm_1/batchnorm/addAdd3current_policy_network/LayerNorm_1/moments/variance2current_policy_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
Я
2current_policy_network/LayerNorm_1/batchnorm/RsqrtRsqrt0current_policy_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
╠
0current_policy_network/LayerNorm_1/batchnorm/mulMul2current_policy_network/LayerNorm_1/batchnorm/Rsqrt-current_policy_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
╗
2current_policy_network/LayerNorm_1/batchnorm/mul_1Mulcurrent_policy_network/add_10current_policy_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
╬
2current_policy_network/LayerNorm_1/batchnorm/mul_2Mul/current_policy_network/LayerNorm_1/moments/mean0current_policy_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
╦
0current_policy_network/LayerNorm_1/batchnorm/subSub,current_policy_network/LayerNorm_1/beta/read2current_policy_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╤
2current_policy_network/LayerNorm_1/batchnorm/add_1Add2current_policy_network/LayerNorm_1/batchnorm/mul_10current_policy_network/LayerNorm_1/batchnorm/sub*
T0*'
_output_shapes
:         @
Л
current_policy_network/Tanh_1Tanh2current_policy_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
э
Tcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/shapeConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
▀
Rcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB
 *═╠╠╜*
dtype0
▀
Rcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
█
\current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformTcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/shape*

seed*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
seed2Э*
dtype0*
_output_shapes

:@
ъ
Rcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/subSubRcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxRcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
_output_shapes
: 
№
Rcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulMul\current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformRcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ю
Ncurrent_policy_network/current_policy_network/out/w/Initializer/random_uniformAddRcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulRcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w
я
3current_policy_network/current_policy_network/out/w
VariableV2*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
у
:current_policy_network/current_policy_network/out/w/AssignAssign3current_policy_network/current_policy_network/out/wNcurrent_policy_network/current_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
ъ
8current_policy_network/current_policy_network/out/w/readIdentity3current_policy_network/current_policy_network/out/w*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
_output_shapes

:@
┌
Ecurrent_policy_network/current_policy_network/out/b/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ч
3current_policy_network/current_policy_network/out/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
	container 
╓
:current_policy_network/current_policy_network/out/b/AssignAssign3current_policy_network/current_policy_network/out/bEcurrent_policy_network/current_policy_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b
ц
8current_policy_network/current_policy_network/out/b/readIdentity3current_policy_network/current_policy_network/out/b*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
_output_shapes
:*
T0
┌
current_policy_network/MatMul_2MatMulcurrent_policy_network/Tanh_18current_policy_network/current_policy_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
░
current_policy_network/add_2Addcurrent_policy_network/MatMul_28current_policy_network/current_policy_network/out/b/read*
T0*'
_output_shapes
:         
щ
Rtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
valueB"   @   
█
Ptarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
█
Ptarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╒
Ztarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/shape*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
seed2н*
dtype0*
_output_shapes

:@*

seed*
T0
т
Ptarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/subSubPtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/maxPtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
_output_shapes
: 
Ї
Ptarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/mulMulZtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
_output_shapes

:@
ц
Ltarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniformAddPtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/mulPtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
_output_shapes

:@
ы
1target_policy_network/target_policy_network/fc0/w
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w
█
8target_policy_network/target_policy_network/fc0/w/AssignAssign1target_policy_network/target_policy_network/fc0/wLtarget_policy_network/target_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
ф
6target_policy_network/target_policy_network/fc0/w/readIdentity1target_policy_network/target_policy_network/fc0/w*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
_output_shapes

:@
╓
Ctarget_policy_network/target_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1target_policy_network/target_policy_network/fc0/b
VariableV2*
_output_shapes
:@*
shared_name *D
_class:
86loc:@target_policy_network/target_policy_network/fc0/b*
	container *
shape:@*
dtype0
╬
8target_policy_network/target_policy_network/fc0/b/AssignAssign1target_policy_network/target_policy_network/fc0/bCtarget_policy_network/target_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/b*
validate_shape(
р
6target_policy_network/target_policy_network/fc0/b/readIdentity1target_policy_network/target_policy_network/fc0/b*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/b*
_output_shapes
:@
╞
target_policy_network/MatMulMatMulobservations_26target_policy_network/target_policy_network/fc0/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
и
target_policy_network/addAddtarget_policy_network/MatMul6target_policy_network/target_policy_network/fc0/b/read*
T0*'
_output_shapes
:         @
╝
6target_policy_network/LayerNorm/beta/Initializer/zerosConst*7
_class-
+)loc:@target_policy_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╔
$target_policy_network/LayerNorm/beta
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@target_policy_network/LayerNorm/beta
Ъ
+target_policy_network/LayerNorm/beta/AssignAssign$target_policy_network/LayerNorm/beta6target_policy_network/LayerNorm/beta/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*7
_class-
+)loc:@target_policy_network/LayerNorm/beta*
validate_shape(
╣
)target_policy_network/LayerNorm/beta/readIdentity$target_policy_network/LayerNorm/beta*7
_class-
+)loc:@target_policy_network/LayerNorm/beta*
_output_shapes
:@*
T0
╜
6target_policy_network/LayerNorm/gamma/Initializer/onesConst*8
_class.
,*loc:@target_policy_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╦
%target_policy_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@target_policy_network/LayerNorm/gamma*
	container *
shape:@
Э
,target_policy_network/LayerNorm/gamma/AssignAssign%target_policy_network/LayerNorm/gamma6target_policy_network/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@target_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
╝
*target_policy_network/LayerNorm/gamma/readIdentity%target_policy_network/LayerNorm/gamma*
T0*8
_class.
,*loc:@target_policy_network/LayerNorm/gamma*
_output_shapes
:@
И
>target_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
▐
,target_policy_network/LayerNorm/moments/meanMeantarget_policy_network/add>target_policy_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
д
4target_policy_network/LayerNorm/moments/StopGradientStopGradient,target_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:         
╤
9target_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencetarget_policy_network/add4target_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:         @
М
Btarget_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ж
0target_policy_network/LayerNorm/moments/varianceMean9target_policy_network/LayerNorm/moments/SquaredDifferenceBtarget_policy_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
t
/target_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╔
-target_policy_network/LayerNorm/batchnorm/addAdd0target_policy_network/LayerNorm/moments/variance/target_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
Щ
/target_policy_network/LayerNorm/batchnorm/RsqrtRsqrt-target_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
├
-target_policy_network/LayerNorm/batchnorm/mulMul/target_policy_network/LayerNorm/batchnorm/Rsqrt*target_policy_network/LayerNorm/gamma/read*'
_output_shapes
:         @*
T0
▓
/target_policy_network/LayerNorm/batchnorm/mul_1Multarget_policy_network/add-target_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
┼
/target_policy_network/LayerNorm/batchnorm/mul_2Mul,target_policy_network/LayerNorm/moments/mean-target_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┬
-target_policy_network/LayerNorm/batchnorm/subSub)target_policy_network/LayerNorm/beta/read/target_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╚
/target_policy_network/LayerNorm/batchnorm/add_1Add/target_policy_network/LayerNorm/batchnorm/mul_1-target_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:         @
Е
target_policy_network/TanhTanh/target_policy_network/LayerNorm/batchnorm/add_1*'
_output_shapes
:         @*
T0
щ
Rtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
█
Ptarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
valueB
 *  А┐*
dtype0
█
Ptarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╒
Ztarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
seed2╘*
dtype0*
_output_shapes

:@@*

seed
т
Ptarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/subSubPtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/maxPtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w
Ї
Ptarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/mulMulZtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
_output_shapes

:@@
ц
Ltarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniformAddPtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/mulPtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
_output_shapes

:@@
ы
1target_policy_network/target_policy_network/fc1/w
VariableV2*
shared_name *D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
█
8target_policy_network/target_policy_network/fc1/w/AssignAssign1target_policy_network/target_policy_network/fc1/wLtarget_policy_network/target_policy_network/fc1/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w
ф
6target_policy_network/target_policy_network/fc1/w/readIdentity1target_policy_network/target_policy_network/fc1/w*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
_output_shapes

:@@*
T0
╓
Ctarget_policy_network/target_policy_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1target_policy_network/target_policy_network/fc1/b
VariableV2*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
╬
8target_policy_network/target_policy_network/fc1/b/AssignAssign1target_policy_network/target_policy_network/fc1/bCtarget_policy_network/target_policy_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/b*
validate_shape(
р
6target_policy_network/target_policy_network/fc1/b/readIdentity1target_policy_network/target_policy_network/fc1/b*
_output_shapes
:@*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/b
╘
target_policy_network/MatMul_1MatMultarget_policy_network/Tanh6target_policy_network/target_policy_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
м
target_policy_network/add_1Addtarget_policy_network/MatMul_16target_policy_network/target_policy_network/fc1/b/read*'
_output_shapes
:         @*
T0
└
8target_policy_network/LayerNorm_1/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*9
_class/
-+loc:@target_policy_network/LayerNorm_1/beta*
valueB@*    
═
&target_policy_network/LayerNorm_1/beta
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@target_policy_network/LayerNorm_1/beta
в
-target_policy_network/LayerNorm_1/beta/AssignAssign&target_policy_network/LayerNorm_1/beta8target_policy_network/LayerNorm_1/beta/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*9
_class/
-+loc:@target_policy_network/LayerNorm_1/beta*
validate_shape(
┐
+target_policy_network/LayerNorm_1/beta/readIdentity&target_policy_network/LayerNorm_1/beta*9
_class/
-+loc:@target_policy_network/LayerNorm_1/beta*
_output_shapes
:@*
T0
┴
8target_policy_network/LayerNorm_1/gamma/Initializer/onesConst*:
_class0
.,loc:@target_policy_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╧
'target_policy_network/LayerNorm_1/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@target_policy_network/LayerNorm_1/gamma*
	container *
shape:@
е
.target_policy_network/LayerNorm_1/gamma/AssignAssign'target_policy_network/LayerNorm_1/gamma8target_policy_network/LayerNorm_1/gamma/Initializer/ones*
T0*:
_class0
.,loc:@target_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
┬
,target_policy_network/LayerNorm_1/gamma/readIdentity'target_policy_network/LayerNorm_1/gamma*
T0*:
_class0
.,loc:@target_policy_network/LayerNorm_1/gamma*
_output_shapes
:@
К
@target_policy_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ф
.target_policy_network/LayerNorm_1/moments/meanMeantarget_policy_network/add_1@target_policy_network/LayerNorm_1/moments/mean/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
и
6target_policy_network/LayerNorm_1/moments/StopGradientStopGradient.target_policy_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
╫
;target_policy_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencetarget_policy_network/add_16target_policy_network/LayerNorm_1/moments/StopGradient*'
_output_shapes
:         @*
T0
О
Dtarget_policy_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
М
2target_policy_network/LayerNorm_1/moments/varianceMean;target_policy_network/LayerNorm_1/moments/SquaredDifferenceDtarget_policy_network/LayerNorm_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
v
1target_policy_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╧
/target_policy_network/LayerNorm_1/batchnorm/addAdd2target_policy_network/LayerNorm_1/moments/variance1target_policy_network/LayerNorm_1/batchnorm/add/y*'
_output_shapes
:         *
T0
Э
1target_policy_network/LayerNorm_1/batchnorm/RsqrtRsqrt/target_policy_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
╔
/target_policy_network/LayerNorm_1/batchnorm/mulMul1target_policy_network/LayerNorm_1/batchnorm/Rsqrt,target_policy_network/LayerNorm_1/gamma/read*'
_output_shapes
:         @*
T0
╕
1target_policy_network/LayerNorm_1/batchnorm/mul_1Multarget_policy_network/add_1/target_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
╦
1target_policy_network/LayerNorm_1/batchnorm/mul_2Mul.target_policy_network/LayerNorm_1/moments/mean/target_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
╚
/target_policy_network/LayerNorm_1/batchnorm/subSub+target_policy_network/LayerNorm_1/beta/read1target_policy_network/LayerNorm_1/batchnorm/mul_2*'
_output_shapes
:         @*
T0
╬
1target_policy_network/LayerNorm_1/batchnorm/add_1Add1target_policy_network/LayerNorm_1/batchnorm/mul_1/target_policy_network/LayerNorm_1/batchnorm/sub*'
_output_shapes
:         @*
T0
Й
target_policy_network/Tanh_1Tanh1target_policy_network/LayerNorm_1/batchnorm/add_1*'
_output_shapes
:         @*
T0
щ
Rtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
█
Ptarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/minConst*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
valueB
 *═╠╠╜*
dtype0*
_output_shapes
: 
█
Ptarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
valueB
 *═╠╠=*
dtype0
╒
Ztarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
seed2√
т
Ptarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/subSubPtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/maxPtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w
Ї
Ptarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/mulMulZtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/RandomUniformPtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w
ц
Ltarget_policy_network/target_policy_network/out/w/Initializer/random_uniformAddPtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/mulPtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
_output_shapes

:@
ы
1target_policy_network/target_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
	container *
shape
:@
█
8target_policy_network/target_policy_network/out/w/AssignAssign1target_policy_network/target_policy_network/out/wLtarget_policy_network/target_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
validate_shape(*
_output_shapes

:@
ф
6target_policy_network/target_policy_network/out/w/readIdentity1target_policy_network/target_policy_network/out/w*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
_output_shapes

:@
╓
Ctarget_policy_network/target_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@target_policy_network/target_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
у
1target_policy_network/target_policy_network/out/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@target_policy_network/target_policy_network/out/b*
	container 
╬
8target_policy_network/target_policy_network/out/b/AssignAssign1target_policy_network/target_policy_network/out/bCtarget_policy_network/target_policy_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/b*
validate_shape(
р
6target_policy_network/target_policy_network/out/b/readIdentity1target_policy_network/target_policy_network/out/b*
_output_shapes
:*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/b
╓
target_policy_network/MatMul_2MatMultarget_policy_network/Tanh_16target_policy_network/target_policy_network/out/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
м
target_policy_network/add_2Addtarget_policy_network/MatMul_26target_policy_network/target_policy_network/out/b/read*'
_output_shapes
:         *
T0
с
Nlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
╙
Llast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/minConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╙
Llast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
valueB
 *  А?
╔
Vlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformNlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
seed2Л
╥
Llast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/subSubLlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxLlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
_output_shapes
: 
ф
Llast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulMulVlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformLlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/sub*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@*
T0
╓
Hlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniformAddLlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulLlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w
у
-last_policy_network/last_policy_network/fc0/w
VariableV2*
_output_shapes

:@*
shared_name *@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
	container *
shape
:@*
dtype0
╦
4last_policy_network/last_policy_network/fc0/w/AssignAssign-last_policy_network/last_policy_network/fc0/wHlast_policy_network/last_policy_network/fc0/w/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
validate_shape(
╪
2last_policy_network/last_policy_network/fc0/w/readIdentity-last_policy_network/last_policy_network/fc0/w*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
╬
?last_policy_network/last_policy_network/fc0/b/Initializer/zerosConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
█
-last_policy_network/last_policy_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@last_policy_network/last_policy_network/fc0/b*
	container *
shape:@
╛
4last_policy_network/last_policy_network/fc0/b/AssignAssign-last_policy_network/last_policy_network/fc0/b?last_policy_network/last_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/b*
validate_shape(
╘
2last_policy_network/last_policy_network/fc0/b/readIdentity-last_policy_network/last_policy_network/fc0/b*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/b*
_output_shapes
:@
└
last_policy_network/MatMulMatMulobservations_22last_policy_network/last_policy_network/fc0/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
а
last_policy_network/addAddlast_policy_network/MatMul2last_policy_network/last_policy_network/fc0/b/read*'
_output_shapes
:         @*
T0
╕
4last_policy_network/LayerNorm/beta/Initializer/zerosConst*5
_class+
)'loc:@last_policy_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
┼
"last_policy_network/LayerNorm/beta
VariableV2*
shared_name *5
_class+
)'loc:@last_policy_network/LayerNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
Т
)last_policy_network/LayerNorm/beta/AssignAssign"last_policy_network/LayerNorm/beta4last_policy_network/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@last_policy_network/LayerNorm/beta
│
'last_policy_network/LayerNorm/beta/readIdentity"last_policy_network/LayerNorm/beta*
T0*5
_class+
)'loc:@last_policy_network/LayerNorm/beta*
_output_shapes
:@
╣
4last_policy_network/LayerNorm/gamma/Initializer/onesConst*6
_class,
*(loc:@last_policy_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╟
#last_policy_network/LayerNorm/gamma
VariableV2*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@last_policy_network/LayerNorm/gamma*
	container *
shape:@*
dtype0
Х
*last_policy_network/LayerNorm/gamma/AssignAssign#last_policy_network/LayerNorm/gamma4last_policy_network/LayerNorm/gamma/Initializer/ones*
_output_shapes
:@*
use_locking(*
T0*6
_class,
*(loc:@last_policy_network/LayerNorm/gamma*
validate_shape(
╢
(last_policy_network/LayerNorm/gamma/readIdentity#last_policy_network/LayerNorm/gamma*
_output_shapes
:@*
T0*6
_class,
*(loc:@last_policy_network/LayerNorm/gamma
Ж
<last_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╪
*last_policy_network/LayerNorm/moments/meanMeanlast_policy_network/add<last_policy_network/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
а
2last_policy_network/LayerNorm/moments/StopGradientStopGradient*last_policy_network/LayerNorm/moments/mean*
T0*'
_output_shapes
:         
╦
7last_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencelast_policy_network/add2last_policy_network/LayerNorm/moments/StopGradient*'
_output_shapes
:         @*
T0
К
@last_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
А
.last_policy_network/LayerNorm/moments/varianceMean7last_policy_network/LayerNorm/moments/SquaredDifference@last_policy_network/LayerNorm/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
r
-last_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
├
+last_policy_network/LayerNorm/batchnorm/addAdd.last_policy_network/LayerNorm/moments/variance-last_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
Х
-last_policy_network/LayerNorm/batchnorm/RsqrtRsqrt+last_policy_network/LayerNorm/batchnorm/add*'
_output_shapes
:         *
T0
╜
+last_policy_network/LayerNorm/batchnorm/mulMul-last_policy_network/LayerNorm/batchnorm/Rsqrt(last_policy_network/LayerNorm/gamma/read*
T0*'
_output_shapes
:         @
м
-last_policy_network/LayerNorm/batchnorm/mul_1Mullast_policy_network/add+last_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┐
-last_policy_network/LayerNorm/batchnorm/mul_2Mul*last_policy_network/LayerNorm/moments/mean+last_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
╝
+last_policy_network/LayerNorm/batchnorm/subSub'last_policy_network/LayerNorm/beta/read-last_policy_network/LayerNorm/batchnorm/mul_2*'
_output_shapes
:         @*
T0
┬
-last_policy_network/LayerNorm/batchnorm/add_1Add-last_policy_network/LayerNorm/batchnorm/mul_1+last_policy_network/LayerNorm/batchnorm/sub*'
_output_shapes
:         @*
T0
Б
last_policy_network/TanhTanh-last_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:         @
с
Nlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
╙
Llast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/minConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╙
Llast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
Vlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformNlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
seed2▓
╥
Llast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/subSubLlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxLlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
_output_shapes
: *
T0
ф
Llast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulMulVlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformLlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
╓
Hlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniformAddLlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulLlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
у
-last_policy_network/last_policy_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
	container *
shape
:@@
╦
4last_policy_network/last_policy_network/fc1/w/AssignAssign-last_policy_network/last_policy_network/fc1/wHlast_policy_network/last_policy_network/fc1/w/Initializer/random_uniform*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
╪
2last_policy_network/last_policy_network/fc1/w/readIdentity-last_policy_network/last_policy_network/fc1/w*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
╬
?last_policy_network/last_policy_network/fc1/b/Initializer/zerosConst*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
█
-last_policy_network/last_policy_network/fc1/b
VariableV2*
shared_name *@
_class6
42loc:@last_policy_network/last_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
╛
4last_policy_network/last_policy_network/fc1/b/AssignAssign-last_policy_network/last_policy_network/fc1/b?last_policy_network/last_policy_network/fc1/b/Initializer/zeros*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
╘
2last_policy_network/last_policy_network/fc1/b/readIdentity-last_policy_network/last_policy_network/fc1/b*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/b*
_output_shapes
:@
╠
last_policy_network/MatMul_1MatMullast_policy_network/Tanh2last_policy_network/last_policy_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
д
last_policy_network/add_1Addlast_policy_network/MatMul_12last_policy_network/last_policy_network/fc1/b/read*
T0*'
_output_shapes
:         @
╝
6last_policy_network/LayerNorm_1/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*7
_class-
+)loc:@last_policy_network/LayerNorm_1/beta*
valueB@*    
╔
$last_policy_network/LayerNorm_1/beta
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *7
_class-
+)loc:@last_policy_network/LayerNorm_1/beta*
	container 
Ъ
+last_policy_network/LayerNorm_1/beta/AssignAssign$last_policy_network/LayerNorm_1/beta6last_policy_network/LayerNorm_1/beta/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@last_policy_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
╣
)last_policy_network/LayerNorm_1/beta/readIdentity$last_policy_network/LayerNorm_1/beta*
T0*7
_class-
+)loc:@last_policy_network/LayerNorm_1/beta*
_output_shapes
:@
╜
6last_policy_network/LayerNorm_1/gamma/Initializer/onesConst*8
_class.
,*loc:@last_policy_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╦
%last_policy_network/LayerNorm_1/gamma
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@last_policy_network/LayerNorm_1/gamma*
	container 
Э
,last_policy_network/LayerNorm_1/gamma/AssignAssign%last_policy_network/LayerNorm_1/gamma6last_policy_network/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@last_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╝
*last_policy_network/LayerNorm_1/gamma/readIdentity%last_policy_network/LayerNorm_1/gamma*
T0*8
_class.
,*loc:@last_policy_network/LayerNorm_1/gamma*
_output_shapes
:@
И
>last_policy_network/LayerNorm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
▐
,last_policy_network/LayerNorm_1/moments/meanMeanlast_policy_network/add_1>last_policy_network/LayerNorm_1/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
д
4last_policy_network/LayerNorm_1/moments/StopGradientStopGradient,last_policy_network/LayerNorm_1/moments/mean*'
_output_shapes
:         *
T0
╤
9last_policy_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencelast_policy_network/add_14last_policy_network/LayerNorm_1/moments/StopGradient*
T0*'
_output_shapes
:         @
М
Blast_policy_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ж
0last_policy_network/LayerNorm_1/moments/varianceMean9last_policy_network/LayerNorm_1/moments/SquaredDifferenceBlast_policy_network/LayerNorm_1/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
t
/last_policy_network/LayerNorm_1/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
╔
-last_policy_network/LayerNorm_1/batchnorm/addAdd0last_policy_network/LayerNorm_1/moments/variance/last_policy_network/LayerNorm_1/batchnorm/add/y*
T0*'
_output_shapes
:         
Щ
/last_policy_network/LayerNorm_1/batchnorm/RsqrtRsqrt-last_policy_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
├
-last_policy_network/LayerNorm_1/batchnorm/mulMul/last_policy_network/LayerNorm_1/batchnorm/Rsqrt*last_policy_network/LayerNorm_1/gamma/read*'
_output_shapes
:         @*
T0
▓
/last_policy_network/LayerNorm_1/batchnorm/mul_1Mullast_policy_network/add_1-last_policy_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
┼
/last_policy_network/LayerNorm_1/batchnorm/mul_2Mul,last_policy_network/LayerNorm_1/moments/mean-last_policy_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
┬
-last_policy_network/LayerNorm_1/batchnorm/subSub)last_policy_network/LayerNorm_1/beta/read/last_policy_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╚
/last_policy_network/LayerNorm_1/batchnorm/add_1Add/last_policy_network/LayerNorm_1/batchnorm/mul_1-last_policy_network/LayerNorm_1/batchnorm/sub*
T0*'
_output_shapes
:         @
Е
last_policy_network/Tanh_1Tanh/last_policy_network/LayerNorm_1/batchnorm/add_1*
T0*'
_output_shapes
:         @
с
Nlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
╙
Llast_policy_network/last_policy_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
valueB
 *═╠╠╜
╙
Llast_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
╔
Vlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformNlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
seed2┘*
dtype0*
_output_shapes

:@*

seed
╥
Llast_policy_network/last_policy_network/out/w/Initializer/random_uniform/subSubLlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxLlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
_output_shapes
: 
ф
Llast_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulMulVlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformLlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
_output_shapes

:@
╓
Hlast_policy_network/last_policy_network/out/w/Initializer/random_uniformAddLlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulLlast_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
_output_shapes

:@
у
-last_policy_network/last_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
	container *
shape
:@
╦
4last_policy_network/last_policy_network/out/w/AssignAssign-last_policy_network/last_policy_network/out/wHlast_policy_network/last_policy_network/out/w/Initializer/random_uniform*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
╪
2last_policy_network/last_policy_network/out/w/readIdentity-last_policy_network/last_policy_network/out/w*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
_output_shapes

:@
╬
?last_policy_network/last_policy_network/out/b/Initializer/zerosConst*@
_class6
42loc:@last_policy_network/last_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
█
-last_policy_network/last_policy_network/out/b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *@
_class6
42loc:@last_policy_network/last_policy_network/out/b*
	container 
╛
4last_policy_network/last_policy_network/out/b/AssignAssign-last_policy_network/last_policy_network/out/b?last_policy_network/last_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:
╘
2last_policy_network/last_policy_network/out/b/readIdentity-last_policy_network/last_policy_network/out/b*@
_class6
42loc:@last_policy_network/last_policy_network/out/b*
_output_shapes
:*
T0
╬
last_policy_network/MatMul_2MatMullast_policy_network/Tanh_12last_policy_network/last_policy_network/out/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
д
last_policy_network/add_2Addlast_policy_network/MatMul_22last_policy_network/last_policy_network/out/b/read*'
_output_shapes
:         *
T0
с
Nbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
╙
Lbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/minConst*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
valueB
 *  А┐*
dtype0*
_output_shapes
: 
╙
Lbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
Vbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformNbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
seed2щ
╥
Lbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubLbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxLbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
_output_shapes
: 
ф
Lbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulVbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformLbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
╓
Hbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddLbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulLbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
у
-best_policy_network/best_policy_network/fc0/w
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w
╦
4best_policy_network/best_policy_network/fc0/w/AssignAssign-best_policy_network/best_policy_network/fc0/wHbest_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
╪
2best_policy_network/best_policy_network/fc0/w/readIdentity-best_policy_network/best_policy_network/fc0/w*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
╬
?best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*
_output_shapes
:@*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/b*
valueB@*    *
dtype0
█
-best_policy_network/best_policy_network/fc0/b
VariableV2*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
╛
4best_policy_network/best_policy_network/fc0/b/AssignAssign-best_policy_network/best_policy_network/fc0/b?best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
╘
2best_policy_network/best_policy_network/fc0/b/readIdentity-best_policy_network/best_policy_network/fc0/b*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/b*
_output_shapes
:@
└
best_policy_network/MatMulMatMulobservations_22best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
а
best_policy_network/addAddbest_policy_network/MatMul2best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:         @
╕
4best_policy_network/LayerNorm/beta/Initializer/zerosConst*
_output_shapes
:@*5
_class+
)'loc:@best_policy_network/LayerNorm/beta*
valueB@*    *
dtype0
┼
"best_policy_network/LayerNorm/beta
VariableV2*
shared_name *5
_class+
)'loc:@best_policy_network/LayerNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
Т
)best_policy_network/LayerNorm/beta/AssignAssign"best_policy_network/LayerNorm/beta4best_policy_network/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
│
'best_policy_network/LayerNorm/beta/readIdentity"best_policy_network/LayerNorm/beta*
T0*5
_class+
)'loc:@best_policy_network/LayerNorm/beta*
_output_shapes
:@
╣
4best_policy_network/LayerNorm/gamma/Initializer/onesConst*6
_class,
*(loc:@best_policy_network/LayerNorm/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╟
#best_policy_network/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@best_policy_network/LayerNorm/gamma*
	container *
shape:@
Х
*best_policy_network/LayerNorm/gamma/AssignAssign#best_policy_network/LayerNorm/gamma4best_policy_network/LayerNorm/gamma/Initializer/ones*6
_class,
*(loc:@best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
╢
(best_policy_network/LayerNorm/gamma/readIdentity#best_policy_network/LayerNorm/gamma*6
_class,
*(loc:@best_policy_network/LayerNorm/gamma*
_output_shapes
:@*
T0
Ж
<best_policy_network/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
╪
*best_policy_network/LayerNorm/moments/meanMeanbest_policy_network/add<best_policy_network/LayerNorm/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
а
2best_policy_network/LayerNorm/moments/StopGradientStopGradient*best_policy_network/LayerNorm/moments/mean*'
_output_shapes
:         *
T0
╦
7best_policy_network/LayerNorm/moments/SquaredDifferenceSquaredDifferencebest_policy_network/add2best_policy_network/LayerNorm/moments/StopGradient*
T0*'
_output_shapes
:         @
К
@best_policy_network/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
А
.best_policy_network/LayerNorm/moments/varianceMean7best_policy_network/LayerNorm/moments/SquaredDifference@best_policy_network/LayerNorm/moments/variance/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
r
-best_policy_network/LayerNorm/batchnorm/add/yConst*
valueB
 *╠╝М+*
dtype0*
_output_shapes
: 
├
+best_policy_network/LayerNorm/batchnorm/addAdd.best_policy_network/LayerNorm/moments/variance-best_policy_network/LayerNorm/batchnorm/add/y*
T0*'
_output_shapes
:         
Х
-best_policy_network/LayerNorm/batchnorm/RsqrtRsqrt+best_policy_network/LayerNorm/batchnorm/add*
T0*'
_output_shapes
:         
╜
+best_policy_network/LayerNorm/batchnorm/mulMul-best_policy_network/LayerNorm/batchnorm/Rsqrt(best_policy_network/LayerNorm/gamma/read*'
_output_shapes
:         @*
T0
м
-best_policy_network/LayerNorm/batchnorm/mul_1Mulbest_policy_network/add+best_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
┐
-best_policy_network/LayerNorm/batchnorm/mul_2Mul*best_policy_network/LayerNorm/moments/mean+best_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
╝
+best_policy_network/LayerNorm/batchnorm/subSub'best_policy_network/LayerNorm/beta/read-best_policy_network/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:         @
┬
-best_policy_network/LayerNorm/batchnorm/add_1Add-best_policy_network/LayerNorm/batchnorm/mul_1+best_policy_network/LayerNorm/batchnorm/sub*
T0*'
_output_shapes
:         @
Б
best_policy_network/TanhTanh-best_policy_network/LayerNorm/batchnorm/add_1*
T0*'
_output_shapes
:         @
с
Nbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
╙
Lbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
valueB
 *  А┐*
dtype0
╙
Lbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
Vbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformNbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shape*

seed*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
seed2Р*
dtype0*
_output_shapes

:@@
╥
Lbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/subSubLbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxLbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
_output_shapes
: 
ф
Lbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulMulVbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformLbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w
╓
Hbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniformAddLbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulLbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
у
-best_policy_network/best_policy_network/fc1/w
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w
╦
4best_policy_network/best_policy_network/fc1/w/AssignAssign-best_policy_network/best_policy_network/fc1/wHbest_policy_network/best_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
╪
2best_policy_network/best_policy_network/fc1/w/readIdentity-best_policy_network/best_policy_network/fc1/w*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
╬
?best_policy_network/best_policy_network/fc1/b/Initializer/zerosConst*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
█
-best_policy_network/best_policy_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@best_policy_network/best_policy_network/fc1/b*
	container *
shape:@
╛
4best_policy_network/best_policy_network/fc1/b/AssignAssign-best_policy_network/best_policy_network/fc1/b?best_policy_network/best_policy_network/fc1/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/b
╘
2best_policy_network/best_policy_network/fc1/b/readIdentity-best_policy_network/best_policy_network/fc1/b*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/b*
_output_shapes
:@
╠
best_policy_network/MatMul_1MatMulbest_policy_network/Tanh2best_policy_network/best_policy_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
д
best_policy_network/add_1Addbest_policy_network/MatMul_12best_policy_network/best_policy_network/fc1/b/read*
T0*'
_output_shapes
:         @
╝
6best_policy_network/LayerNorm_1/beta/Initializer/zerosConst*
_output_shapes
:@*7
_class-
+)loc:@best_policy_network/LayerNorm_1/beta*
valueB@*    *
dtype0
╔
$best_policy_network/LayerNorm_1/beta
VariableV2*
shared_name *7
_class-
+)loc:@best_policy_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ъ
+best_policy_network/LayerNorm_1/beta/AssignAssign$best_policy_network/LayerNorm_1/beta6best_policy_network/LayerNorm_1/beta/Initializer/zeros*
T0*7
_class-
+)loc:@best_policy_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
╣
)best_policy_network/LayerNorm_1/beta/readIdentity$best_policy_network/LayerNorm_1/beta*
T0*7
_class-
+)loc:@best_policy_network/LayerNorm_1/beta*
_output_shapes
:@
╜
6best_policy_network/LayerNorm_1/gamma/Initializer/onesConst*8
_class.
,*loc:@best_policy_network/LayerNorm_1/gamma*
valueB@*  А?*
dtype0*
_output_shapes
:@
╦
%best_policy_network/LayerNorm_1/gamma
VariableV2*8
_class.
,*loc:@best_policy_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Э
,best_policy_network/LayerNorm_1/gamma/AssignAssign%best_policy_network/LayerNorm_1/gamma6best_policy_network/LayerNorm_1/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*8
_class.
,*loc:@best_policy_network/LayerNorm_1/gamma
╝
*best_policy_network/LayerNorm_1/gamma/readIdentity%best_policy_network/LayerNorm_1/gamma*
T0*8
_class.
,*loc:@best_policy_network/LayerNorm_1/gamma*
_output_shapes
:@
И
>best_policy_network/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
▐
,best_policy_network/LayerNorm_1/moments/meanMeanbest_policy_network/add_1>best_policy_network/LayerNorm_1/moments/mean/reduction_indices*'
_output_shapes
:         *
	keep_dims(*

Tidx0*
T0
д
4best_policy_network/LayerNorm_1/moments/StopGradientStopGradient,best_policy_network/LayerNorm_1/moments/mean*
T0*'
_output_shapes
:         
╤
9best_policy_network/LayerNorm_1/moments/SquaredDifferenceSquaredDifferencebest_policy_network/add_14best_policy_network/LayerNorm_1/moments/StopGradient*
T0*'
_output_shapes
:         @
М
Bbest_policy_network/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ж
0best_policy_network/LayerNorm_1/moments/varianceMean9best_policy_network/LayerNorm_1/moments/SquaredDifferenceBbest_policy_network/LayerNorm_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
t
/best_policy_network/LayerNorm_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *╠╝М+*
dtype0
╔
-best_policy_network/LayerNorm_1/batchnorm/addAdd0best_policy_network/LayerNorm_1/moments/variance/best_policy_network/LayerNorm_1/batchnorm/add/y*'
_output_shapes
:         *
T0
Щ
/best_policy_network/LayerNorm_1/batchnorm/RsqrtRsqrt-best_policy_network/LayerNorm_1/batchnorm/add*
T0*'
_output_shapes
:         
├
-best_policy_network/LayerNorm_1/batchnorm/mulMul/best_policy_network/LayerNorm_1/batchnorm/Rsqrt*best_policy_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
▓
/best_policy_network/LayerNorm_1/batchnorm/mul_1Mulbest_policy_network/add_1-best_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
┼
/best_policy_network/LayerNorm_1/batchnorm/mul_2Mul,best_policy_network/LayerNorm_1/moments/mean-best_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
┬
-best_policy_network/LayerNorm_1/batchnorm/subSub)best_policy_network/LayerNorm_1/beta/read/best_policy_network/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:         @
╚
/best_policy_network/LayerNorm_1/batchnorm/add_1Add/best_policy_network/LayerNorm_1/batchnorm/mul_1-best_policy_network/LayerNorm_1/batchnorm/sub*'
_output_shapes
:         @*
T0
Е
best_policy_network/Tanh_1Tanh/best_policy_network/LayerNorm_1/batchnorm/add_1*'
_output_shapes
:         @*
T0
с
Nbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
valueB"@      *
dtype0
╙
Lbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/minConst*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
valueB
 *═╠╠╜*
dtype0*
_output_shapes
: 
╙
Lbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
╔
Vbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformNbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/shape*
seed2╖*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/w
╥
Lbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/subSubLbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/maxLbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
_output_shapes
: 
ф
Lbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulMulVbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/RandomUniformLbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/sub*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
_output_shapes

:@*
T0
╓
Hbest_policy_network/best_policy_network/out/w/Initializer/random_uniformAddLbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulLbest_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
_output_shapes

:@
у
-best_policy_network/best_policy_network/out/w
VariableV2*
_output_shapes

:@*
shared_name *@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
	container *
shape
:@*
dtype0
╦
4best_policy_network/best_policy_network/out/w/AssignAssign-best_policy_network/best_policy_network/out/wHbest_policy_network/best_policy_network/out/w/Initializer/random_uniform*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
╪
2best_policy_network/best_policy_network/out/w/readIdentity-best_policy_network/best_policy_network/out/w*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
_output_shapes

:@
╬
?best_policy_network/best_policy_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@best_policy_network/best_policy_network/out/b*
valueB*    
█
-best_policy_network/best_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *@
_class6
42loc:@best_policy_network/best_policy_network/out/b*
	container *
shape:
╛
4best_policy_network/best_policy_network/out/b/AssignAssign-best_policy_network/best_policy_network/out/b?best_policy_network/best_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
╘
2best_policy_network/best_policy_network/out/b/readIdentity-best_policy_network/best_policy_network/out/b*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/b*
_output_shapes
:
╬
best_policy_network/MatMul_2MatMulbest_policy_network/Tanh_12best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
д
best_policy_network/add_2Addbest_policy_network/MatMul_22best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:         
d
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
f
strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
f
strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
б
strided_sliceStridedSlicecurrent_policy_network/add_2strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:         *
Index0*
T0
X
SqueezeSqueezestrided_slice*
_output_shapes
:*
squeeze_dims
 *
T0
^
Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
j
ReshapeReshapeSqueezeReshape/shape*
T0*
Tshape0*'
_output_shapes
:         
f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
и
strided_slice_1StridedSlicetarget_policy_network/add_2strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:         *
Index0*
T0
\
	Squeeze_1Squeezestrided_slice_1*
_output_shapes
:*
squeeze_dims
 *
T0
`
Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
p
	Reshape_1Reshape	Squeeze_1Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:         
f
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"        
h
strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ж
strided_slice_2StridedSlicelast_policy_network/add_2strided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:         
\
	Squeeze_2Squeezestrided_slice_2*
T0*
_output_shapes
:*
squeeze_dims
 
`
Reshape_2/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
p
	Reshape_2Reshape	Squeeze_2Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
f
strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_3/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
ж
strided_slice_3StridedSlicebest_policy_network/add_2strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:         
\
	Squeeze_3Squeezestrided_slice_3*
squeeze_dims
 *
T0*
_output_shapes
:
`
Reshape_3/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
p
	Reshape_3Reshape	Squeeze_3Reshape_3/shape*
Tshape0*'
_output_shapes
:         *
T0
Q

Normal/locIdentityReshape*
T0*'
_output_shapes
:         
D
Normal/scaleIdentitystd_dev*
_output_shapes
:*
T0
U
Normal_1/locIdentity	Reshape_1*
T0*'
_output_shapes
:         
F
Normal_1/scaleIdentitystd_dev*
T0*
_output_shapes
:
U
Normal_2/locIdentity	Reshape_2*
T0*'
_output_shapes
:         
F
Normal_2/scaleIdentitystd_dev*
_output_shapes
:*
T0
U
Normal_3/locIdentity	Reshape_3*'
_output_shapes
:         *
T0
F
Normal_3/scaleIdentitystd_dev*
_output_shapes
:*
T0
k
&KullbackLeibler/kl_normal_normal/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
m
(KullbackLeibler/kl_normal_normal/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
m
(KullbackLeibler/kl_normal_normal/Const_2Const*
_output_shapes
: *
valueB
 *   ?*
dtype0
b
'KullbackLeibler/kl_normal_normal/SquareSquareNormal/scale*
_output_shapes
:*
T0
f
)KullbackLeibler/kl_normal_normal/Square_1SquareNormal_2/scale*
T0*
_output_shapes
:
к
(KullbackLeibler/kl_normal_normal/truedivRealDiv'KullbackLeibler/kl_normal_normal/Square)KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
:
w
$KullbackLeibler/kl_normal_normal/subSub
Normal/locNormal_2/loc*
T0*'
_output_shapes
:         
Л
)KullbackLeibler/kl_normal_normal/Square_2Square$KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:         
г
$KullbackLeibler/kl_normal_normal/mulMul(KullbackLeibler/kl_normal_normal/Const_1)KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
:*
T0
й
*KullbackLeibler/kl_normal_normal/truediv_1RealDiv)KullbackLeibler/kl_normal_normal/Square_2$KullbackLeibler/kl_normal_normal/mul*
T0*
_output_shapes
:
в
&KullbackLeibler/kl_normal_normal/sub_1Sub(KullbackLeibler/kl_normal_normal/truediv&KullbackLeibler/kl_normal_normal/Const*
T0*
_output_shapes
:
x
$KullbackLeibler/kl_normal_normal/LogLog(KullbackLeibler/kl_normal_normal/truediv*
_output_shapes
:*
T0
Ю
&KullbackLeibler/kl_normal_normal/sub_2Sub&KullbackLeibler/kl_normal_normal/sub_1$KullbackLeibler/kl_normal_normal/Log*
T0*
_output_shapes
:
в
&KullbackLeibler/kl_normal_normal/mul_1Mul(KullbackLeibler/kl_normal_normal/Const_2&KullbackLeibler/kl_normal_normal/sub_2*
T0*
_output_shapes
:
в
$KullbackLeibler/kl_normal_normal/addAdd*KullbackLeibler/kl_normal_normal/truediv_1&KullbackLeibler/kl_normal_normal/mul_1*
_output_shapes
:*
T0
S
RankRank$KullbackLeibler/kl_normal_normal/add*
_output_shapes
: *
T0
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
_
rangeRangerange/startRankrange/delta*#
_output_shapes
:         *

Tidx0
{
Mean_3Mean$KullbackLeibler/kl_normal_normal/addrange*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
m
!Normal_4/batch_shape_tensor/ShapeShapeNormal_1/loc*
T0*
out_type0*
_output_shapes
:
z
#Normal_4/batch_shape_tensor/Shape_1ShapeNormal_1/scale*#
_output_shapes
:         *
T0*
out_type0
░
)Normal_4/batch_shape_tensor/BroadcastArgsBroadcastArgs!Normal_4/batch_shape_tensor/Shape#Normal_4/batch_shape_tensor/Shape_1*#
_output_shapes
:         *
T0
[
concat_2/values_0Const*
valueB:
*
dtype0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
д
concat_2ConcatV2concat_2/values_0)Normal_4/batch_shape_tensor/BroadcastArgsconcat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:         
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
О
"random_normal/RandomStandardNormalRandomStandardNormalconcat_2*
_output_shapes
:*
seed2Д*

seed*
T0*
dtype0
u
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:
^
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes
:
L
mulMulrandom_normalNormal_1/scale*
_output_shapes
:*
T0
@
addAddmulNormal_1/loc*
T0*
_output_shapes
:
d
Reshape_4/shapeConst*
dtype0*
_output_shapes
:*!
valueB"    
      
n
	Reshape_4ReshapeaddReshape_4/shape*+
_output_shapes
:         
*
T0*
Tshape0
m
!Normal_5/batch_shape_tensor/ShapeShapeNormal_3/loc*
out_type0*
_output_shapes
:*
T0
z
#Normal_5/batch_shape_tensor/Shape_1ShapeNormal_3/scale*
T0*
out_type0*#
_output_shapes
:         
░
)Normal_5/batch_shape_tensor/BroadcastArgsBroadcastArgs!Normal_5/batch_shape_tensor/Shape#Normal_5/batch_shape_tensor/Shape_1*#
_output_shapes
:         *
T0
[
concat_3/values_0Const*
valueB:
*
dtype0*
_output_shapes
:
O
concat_3/axisConst*
value	B : *
dtype0*
_output_shapes
: 
д
concat_3ConcatV2concat_3/values_0)Normal_5/batch_shape_tensor/BroadcastArgsconcat_3/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
Y
random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_1/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Р
$random_normal_1/RandomStandardNormalRandomStandardNormalconcat_3*
T0*
dtype0*
_output_shapes
:*
seed2У*

seed
{
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes
:
d
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes
:*
T0
P
mul_1Mulrandom_normal_1Normal_3/scale*
_output_shapes
:*
T0
D
add_1Addmul_1Normal_3/loc*
_output_shapes
:*
T0
d
Reshape_5/shapeConst*!
valueB"    
      *
dtype0*
_output_shapes
:
p
	Reshape_5Reshapeadd_1Reshape_5/shape*
T0*
Tshape0*+
_output_shapes
:         

m
!Normal_6/batch_shape_tensor/ShapeShapeNormal_1/loc*
T0*
out_type0*
_output_shapes
:
z
#Normal_6/batch_shape_tensor/Shape_1ShapeNormal_1/scale*
out_type0*#
_output_shapes
:         *
T0
░
)Normal_6/batch_shape_tensor/BroadcastArgsBroadcastArgs!Normal_6/batch_shape_tensor/Shape#Normal_6/batch_shape_tensor/Shape_1*
T0*#
_output_shapes
:         
[
concat_4/values_0Const*
_output_shapes
:*
valueB:*
dtype0
O
concat_4/axisConst*
_output_shapes
: *
value	B : *
dtype0
д
concat_4ConcatV2concat_4/values_0)Normal_6/batch_shape_tensor/BroadcastArgsconcat_4/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
Y
random_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_2/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Р
$random_normal_2/RandomStandardNormalRandomStandardNormalconcat_4*
T0*
dtype0*
_output_shapes
:*
seed2в*

seed
{
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes
:*
T0
d
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes
:*
T0
P
mul_2Mulrandom_normal_2Normal_1/scale*
_output_shapes
:*
T0
D
add_2Addmul_2Normal_1/loc*
_output_shapes
:*
T0
`
Reshape_6/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
l
	Reshape_6Reshapeadd_2Reshape_6/shape*
T0*
Tshape0*'
_output_shapes
:         
n
SquaredDifference_2SquaredDifferenceReshape	actions_1*'
_output_shapes
:         *
T0
X
Const_3Const*
dtype0*
_output_shapes
:*
valueB"       
j
Mean_4MeanSquaredDifference_2Const_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
T
gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_2/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_2/FillFillgradients_2/Shapegradients_2/Const*
T0*
_output_shapes
: 
v
%gradients_2/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ъ
gradients_2/Mean_4_grad/ReshapeReshapegradients_2/Fill%gradients_2/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
p
gradients_2/Mean_4_grad/ShapeShapeSquaredDifference_2*
T0*
out_type0*
_output_shapes
:
и
gradients_2/Mean_4_grad/TileTilegradients_2/Mean_4_grad/Reshapegradients_2/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
r
gradients_2/Mean_4_grad/Shape_1ShapeSquaredDifference_2*
T0*
out_type0*
_output_shapes
:
b
gradients_2/Mean_4_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
Ы
gradients_2/Mean_4_grad/ConstConst*
valueB: *2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
dtype0*
_output_shapes
:
╓
gradients_2/Mean_4_grad/ProdProdgradients_2/Mean_4_grad/Shape_1gradients_2/Mean_4_grad/Const*
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
_output_shapes
: 
Э
gradients_2/Mean_4_grad/Const_1Const*
valueB: *2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
dtype0*
_output_shapes
:
┌
gradients_2/Mean_4_grad/Prod_1Prodgradients_2/Mean_4_grad/Shape_2gradients_2/Mean_4_grad/Const_1*
T0*2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0
Ч
!gradients_2/Mean_4_grad/Maximum/yConst*
value	B :*2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
dtype0*
_output_shapes
: 
┬
gradients_2/Mean_4_grad/MaximumMaximumgradients_2/Mean_4_grad/Prod_1!gradients_2/Mean_4_grad/Maximum/y*2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
_output_shapes
: *
T0
└
 gradients_2/Mean_4_grad/floordivFloorDivgradients_2/Mean_4_grad/Prodgradients_2/Mean_4_grad/Maximum*
T0*2
_class(
&$loc:@gradients_2/Mean_4_grad/Shape_1*
_output_shapes
: 
v
gradients_2/Mean_4_grad/CastCast gradients_2/Mean_4_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ш
gradients_2/Mean_4_grad/truedivRealDivgradients_2/Mean_4_grad/Tilegradients_2/Mean_4_grad/Cast*
T0*'
_output_shapes
:         
q
*gradients_2/SquaredDifference_2_grad/ShapeShapeReshape*
T0*
out_type0*
_output_shapes
:
u
,gradients_2/SquaredDifference_2_grad/Shape_1Shape	actions_1*
T0*
out_type0*
_output_shapes
:
ъ
:gradients_2/SquaredDifference_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_2/SquaredDifference_2_grad/Shape,gradients_2/SquaredDifference_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Т
+gradients_2/SquaredDifference_2_grad/scalarConst ^gradients_2/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
п
(gradients_2/SquaredDifference_2_grad/mulMul+gradients_2/SquaredDifference_2_grad/scalargradients_2/Mean_4_grad/truediv*
T0*'
_output_shapes
:         
Ч
(gradients_2/SquaredDifference_2_grad/subSubReshape	actions_1 ^gradients_2/Mean_4_grad/truediv*
T0*'
_output_shapes
:         
╖
*gradients_2/SquaredDifference_2_grad/mul_1Mul(gradients_2/SquaredDifference_2_grad/mul(gradients_2/SquaredDifference_2_grad/sub*
T0*'
_output_shapes
:         
╫
(gradients_2/SquaredDifference_2_grad/SumSum*gradients_2/SquaredDifference_2_grad/mul_1:gradients_2/SquaredDifference_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
═
,gradients_2/SquaredDifference_2_grad/ReshapeReshape(gradients_2/SquaredDifference_2_grad/Sum*gradients_2/SquaredDifference_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
█
*gradients_2/SquaredDifference_2_grad/Sum_1Sum*gradients_2/SquaredDifference_2_grad/mul_1<gradients_2/SquaredDifference_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╙
.gradients_2/SquaredDifference_2_grad/Reshape_1Reshape*gradients_2/SquaredDifference_2_grad/Sum_1,gradients_2/SquaredDifference_2_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
С
(gradients_2/SquaredDifference_2_grad/NegNeg.gradients_2/SquaredDifference_2_grad/Reshape_1*'
_output_shapes
:         *
T0
Ч
5gradients_2/SquaredDifference_2_grad/tuple/group_depsNoOp-^gradients_2/SquaredDifference_2_grad/Reshape)^gradients_2/SquaredDifference_2_grad/Neg
в
=gradients_2/SquaredDifference_2_grad/tuple/control_dependencyIdentity,gradients_2/SquaredDifference_2_grad/Reshape6^gradients_2/SquaredDifference_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/SquaredDifference_2_grad/Reshape*'
_output_shapes
:         
Ь
?gradients_2/SquaredDifference_2_grad/tuple/control_dependency_1Identity(gradients_2/SquaredDifference_2_grad/Neg6^gradients_2/SquaredDifference_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_2/SquaredDifference_2_grad/Neg*'
_output_shapes
:         
n
gradients_2/Reshape_grad/ShapeShapeSqueeze*
T0*
out_type0*#
_output_shapes
:         
╗
 gradients_2/Reshape_grad/ReshapeReshape=gradients_2/SquaredDifference_2_grad/tuple/control_dependencygradients_2/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
k
gradients_2/Squeeze_grad/ShapeShapestrided_slice*
T0*
out_type0*
_output_shapes
:
н
 gradients_2/Squeeze_grad/ReshapeReshape gradients_2/Reshape_grad/Reshapegradients_2/Squeeze_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
А
$gradients_2/strided_slice_grad/ShapeShapecurrent_policy_network/add_2*
T0*
out_type0*
_output_shapes
:
ё
/gradients_2/strided_slice_grad/StridedSliceGradStridedSliceGrad$gradients_2/strided_slice_grad/Shapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2 gradients_2/Squeeze_grad/Reshape*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:         
Т
3gradients_2/current_policy_network/add_2_grad/ShapeShapecurrent_policy_network/MatMul_2*
_output_shapes
:*
T0*
out_type0

5gradients_2/current_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
Cgradients_2/current_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_2/current_policy_network/add_2_grad/Shape5gradients_2/current_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ю
1gradients_2/current_policy_network/add_2_grad/SumSum/gradients_2/strided_slice_grad/StridedSliceGradCgradients_2/current_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ш
5gradients_2/current_policy_network/add_2_grad/ReshapeReshape1gradients_2/current_policy_network/add_2_grad/Sum3gradients_2/current_policy_network/add_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Є
3gradients_2/current_policy_network/add_2_grad/Sum_1Sum/gradients_2/strided_slice_grad/StridedSliceGradEgradients_2/current_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
с
7gradients_2/current_policy_network/add_2_grad/Reshape_1Reshape3gradients_2/current_policy_network/add_2_grad/Sum_15gradients_2/current_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
╕
>gradients_2/current_policy_network/add_2_grad/tuple/group_depsNoOp6^gradients_2/current_policy_network/add_2_grad/Reshape8^gradients_2/current_policy_network/add_2_grad/Reshape_1
╞
Fgradients_2/current_policy_network/add_2_grad/tuple/control_dependencyIdentity5gradients_2/current_policy_network/add_2_grad/Reshape?^gradients_2/current_policy_network/add_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/current_policy_network/add_2_grad/Reshape*'
_output_shapes
:         
┐
Hgradients_2/current_policy_network/add_2_grad/tuple/control_dependency_1Identity7gradients_2/current_policy_network/add_2_grad/Reshape_1?^gradients_2/current_policy_network/add_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/current_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
Ы
7gradients_2/current_policy_network/MatMul_2_grad/MatMulMatMulFgradients_2/current_policy_network/add_2_grad/tuple/control_dependency8current_policy_network/current_policy_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:         @*
transpose_a( 
∙
9gradients_2/current_policy_network/MatMul_2_grad/MatMul_1MatMulcurrent_policy_network/Tanh_1Fgradients_2/current_policy_network/add_2_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
┐
Agradients_2/current_policy_network/MatMul_2_grad/tuple/group_depsNoOp8^gradients_2/current_policy_network/MatMul_2_grad/MatMul:^gradients_2/current_policy_network/MatMul_2_grad/MatMul_1
╨
Igradients_2/current_policy_network/MatMul_2_grad/tuple/control_dependencyIdentity7gradients_2/current_policy_network/MatMul_2_grad/MatMulB^gradients_2/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/current_policy_network/MatMul_2_grad/MatMul*'
_output_shapes
:         @
═
Kgradients_2/current_policy_network/MatMul_2_grad/tuple/control_dependency_1Identity9gradients_2/current_policy_network/MatMul_2_grad/MatMul_1B^gradients_2/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_2/current_policy_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
▀
7gradients_2/current_policy_network/Tanh_1_grad/TanhGradTanhGradcurrent_policy_network/Tanh_1Igradients_2/current_policy_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
╗
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ShapeShape2current_policy_network/LayerNorm_1/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
╗
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1Shape0current_policy_network/LayerNorm_1/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
╟
Ygradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ShapeKgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
в
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/SumSum7gradients_2/current_policy_network/Tanh_1_grad/TanhGradYgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
к
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeReshapeGgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/SumIgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:         @*
T0
ж
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Sum7gradients_2/current_policy_network/Tanh_1_grad/TanhGrad[gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
░
Mgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1ReshapeIgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
·
Tgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_depsNoOpL^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeN^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1
Ю
\gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependencyIdentityKgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeU^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @*
T0
д
^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1IdentityMgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1U^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:         @
е
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeShapecurrent_policy_network/add_1*
out_type0*
_output_shapes
:*
T0
╗
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1Shape0current_policy_network/LayerNorm_1/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
╟
Ygradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeKgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Р
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mulMul\gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency0current_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
▓
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/SumSumGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mulYgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeReshapeGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/SumIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
■
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1Mulcurrent_policy_network/add_1\gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
╕
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1SumIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1[gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
░
Mgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1ReshapeIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
·
Tgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_depsNoOpL^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeN^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1
Ю
\gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyIdentityKgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeU^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:         @*
T0
д
^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityMgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1U^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:         @
С
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:
╗
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape_1Shape2current_policy_network/LayerNorm_1/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ShapeIgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┼
Egradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/SumSum^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Wgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ReshapeReshapeEgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/SumGgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape*
_output_shapes
:@*
T0*
Tshape0
╔
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Sum_1Sum^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Ygradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╕
Egradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/NegNegGgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
и
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1ReshapeEgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/NegIgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_depsNoOpJ^gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ReshapeL^gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1
Й
Zgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependencyIdentityIgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ReshapeS^gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape*
_output_shapes
:@
Ь
\gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1S^gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @
╕
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeShape/current_policy_network/LayerNorm_1/moments/mean*
T0*
out_type0*
_output_shapes
:
╗
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1Shape0current_policy_network/LayerNorm_1/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
╟
Ygradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeKgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mulMul\gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_10current_policy_network/LayerNorm_1/batchnorm/mul*'
_output_shapes
:         @*
T0
▓
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/SumSumGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mulYgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeReshapeGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/SumIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
С
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1Mul/current_policy_network/LayerNorm_1/moments/mean\gradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1*'
_output_shapes
:         @*
T0
╕
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1SumIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1[gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
░
Mgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1ReshapeIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
·
Tgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_depsNoOpL^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeN^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1
Ю
\gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyIdentityKgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeU^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:         *
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape
д
^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityMgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1U^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:         @
ї
gradients_2/AddNAddN^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:         @
╣
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ShapeShape2current_policy_network/LayerNorm_1/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
У
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
┴
Wgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ShapeIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┐
Egradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mulMulgradients_2/AddN-current_policy_network/LayerNorm_1/gamma/read*'
_output_shapes
:         @*
T0
м
Egradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/SumSumEgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mulWgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
д
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ReshapeReshapeEgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/SumGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╞
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mul_1Mul2current_policy_network/LayerNorm_1/batchnorm/Rsqrtgradients_2/AddN*'
_output_shapes
:         @*
T0
▓
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Sum_1SumGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mul_1Ygradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Э
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1ReshapeGgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Sum_1Igradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Ї
Rgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_depsNoOpJ^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ReshapeL^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1
Ц
Zgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependencyIdentityIgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ReshapeS^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape*'
_output_shapes
:         
П
\gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1IdentityKgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1S^gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:@*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1
Ь
Mgradients_2/current_policy_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad2current_policy_network/LayerNorm_1/batchnorm/RsqrtZgradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
║
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/ShapeShape3current_policy_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
М
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
┴
Wgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/ShapeIgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
Egradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/SumSumMgradients_2/current_policy_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradWgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
д
Igradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/ReshapeReshapeEgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/SumGgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╕
Ggradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Sum_1SumMgradients_2/current_policy_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradYgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Щ
Kgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1ReshapeGgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Sum_1Igradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Ї
Rgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/group_depsNoOpJ^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/ReshapeL^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1
Ц
Zgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyIdentityIgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/ReshapeS^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape*'
_output_shapes
:         
Л
\gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependency_1IdentityKgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1S^gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
╞
Jgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ShapeShape<current_policy_network/LayerNorm_1/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
ъ
Igradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/SizeConst*
value	B :*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
х
Hgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/addAddEcurrent_policy_network/LayerNorm_1/moments/variance/reduction_indicesIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Size*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:*
T0
э
Hgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/modFloorModHgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/addIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Size*
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
ї
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_1Const*
valueB:*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
ё
Pgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/range/startConst*
value	B : *]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
ё
Pgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/range/deltaConst*
value	B :*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
╔
Jgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/rangeRangePgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/range/startIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/SizePgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
Ё
Ogradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Fill/valueConst*
value	B :*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
Ї
Igradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/FillFillLgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_1Ogradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Fill/value*
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
ж
Rgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/DynamicStitchDynamicStitchJgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/rangeHgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/modJgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ShapeIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Fill*
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
N*#
_output_shapes
:         
я
Ngradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
value	B :*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0
И
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/MaximumMaximumRgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/DynamicStitchNgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum/y*#
_output_shapes
:         *
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
ў
Mgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/floordivFloorDivJgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ShapeLgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum*
_output_shapes
:*
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
╕
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ReshapeReshapeZgradients_2/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyRgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
╗
Igradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/TileTileLgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ReshapeMgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
╚
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2Shape<current_policy_network/LayerNorm_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
┐
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_3Shape3current_policy_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
ї
Jgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ConstConst*
valueB: *_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
К
Igradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ProdProdLgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2Jgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2
ў
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Const_1Const*
valueB: *_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
О
Kgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Prod_1ProdLgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_3Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Const_1*
T0*_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
є
Pgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1/yConst*
value	B :*_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
·
Ngradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1MaximumKgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Prod_1Pgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2
°
Ogradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/floordiv_1FloorDivIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/ProdNgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2
╥
Igradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/CastCastOgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Я
Lgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/truedivRealDivIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/TileIgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
п
Sgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeShapecurrent_policy_network/add_1*
_output_shapes
:*
T0*
out_type0
╠
Ugradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1Shape7current_policy_network/LayerNorm_1/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
х
cgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeUgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ш
Tgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/scalarConstM^gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
о
Qgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mulMulTgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/scalarLgradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
░
Qgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/subSubcurrent_policy_network/add_17current_policy_network/LayerNorm_1/moments/StopGradientM^gradients_2/current_policy_network/LayerNorm_1/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
▓
Sgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1MulQgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mulQgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/sub*'
_output_shapes
:         @*
T0
╥
Qgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/SumSumSgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1cgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╚
Ugradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeReshapeQgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/SumSgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
╓
Sgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1SumSgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1egradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╬
Wgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1ReshapeSgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1Ugradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
у
Qgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/NegNegWgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
Т
^gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_depsNoOpV^gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeR^gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Neg
╞
fgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyIdentityUgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_^gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:         @
└
hgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityQgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Neg_^gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Neg*'
_output_shapes
:         
в
Fgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ShapeShapecurrent_policy_network/add_1*
T0*
out_type0*
_output_shapes
:
т
Egradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/SizeConst*
value	B :*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╒
Dgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/addAddAcurrent_policy_network/LayerNorm_1/moments/mean/reduction_indicesEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Size*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
▌
Dgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/modFloorModDgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/addEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Size*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
э
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_1Const*
valueB:*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
щ
Lgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/range/startConst*
value	B : *Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
щ
Lgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/range/deltaConst*
_output_shapes
: *
value	B :*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0
╡
Fgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/rangeRangeLgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/range/startEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/SizeLgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape
ш
Kgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Fill/valueConst*
value	B :*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
ф
Egradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/FillFillHgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_1Kgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Fill/value*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
О
Ngradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/DynamicStitchDynamicStitchFgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/rangeDgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/modFgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ShapeEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Fill*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
N*#
_output_shapes
:         
ч
Jgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum/yConst*
value	B :*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
°
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/MaximumMaximumNgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/DynamicStitchJgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum/y*#
_output_shapes
:         *
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape
ч
Igradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/floordivFloorDivFgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ShapeHgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
▓
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ReshapeReshape\gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyNgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
Egradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/TileTileHgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ReshapeIgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
д
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2Shapecurrent_policy_network/add_1*
T0*
out_type0*
_output_shapes
:
╖
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_3Shape/current_policy_network/LayerNorm_1/moments/mean*
out_type0*
_output_shapes
:*
T0
э
Fgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ConstConst*
valueB: *[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
·
Egradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ProdProdHgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2Fgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Const*
	keep_dims( *

Tidx0*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
я
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Const_1Const*
valueB: *[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
■
Ggradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Prod_1ProdHgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_3Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
ы
Lgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0
ъ
Jgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1MaximumGgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Prod_1Lgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1/y*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
ш
Kgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/floordiv_1FloorDivEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/ProdJgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
╩
Egradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/CastCastKgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
У
Hgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/truedivRealDivEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/TileEgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/Cast*'
_output_shapes
:         @*
T0
┼
gradients_2/AddN_1AddN\gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyfgradients_2/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyHgradients_2/current_policy_network/LayerNorm_1/moments/mean_grad/truediv*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:         @
Т
3gradients_2/current_policy_network/add_1_grad/ShapeShapecurrent_policy_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

5gradients_2/current_policy_network/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
Е
Cgradients_2/current_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_2/current_policy_network/add_1_grad/Shape5gradients_2/current_policy_network/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╤
1gradients_2/current_policy_network/add_1_grad/SumSumgradients_2/AddN_1Cgradients_2/current_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ш
5gradients_2/current_policy_network/add_1_grad/ReshapeReshape1gradients_2/current_policy_network/add_1_grad/Sum3gradients_2/current_policy_network/add_1_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
╒
3gradients_2/current_policy_network/add_1_grad/Sum_1Sumgradients_2/AddN_1Egradients_2/current_policy_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
с
7gradients_2/current_policy_network/add_1_grad/Reshape_1Reshape3gradients_2/current_policy_network/add_1_grad/Sum_15gradients_2/current_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
╕
>gradients_2/current_policy_network/add_1_grad/tuple/group_depsNoOp6^gradients_2/current_policy_network/add_1_grad/Reshape8^gradients_2/current_policy_network/add_1_grad/Reshape_1
╞
Fgradients_2/current_policy_network/add_1_grad/tuple/control_dependencyIdentity5gradients_2/current_policy_network/add_1_grad/Reshape?^gradients_2/current_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*H
_class>
<:loc:@gradients_2/current_policy_network/add_1_grad/Reshape
┐
Hgradients_2/current_policy_network/add_1_grad/tuple/control_dependency_1Identity7gradients_2/current_policy_network/add_1_grad/Reshape_1?^gradients_2/current_policy_network/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/current_policy_network/add_1_grad/Reshape_1*
_output_shapes
:@
Ы
7gradients_2/current_policy_network/MatMul_1_grad/MatMulMatMulFgradients_2/current_policy_network/add_1_grad/tuple/control_dependency8current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
ў
9gradients_2/current_policy_network/MatMul_1_grad/MatMul_1MatMulcurrent_policy_network/TanhFgradients_2/current_policy_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
┐
Agradients_2/current_policy_network/MatMul_1_grad/tuple/group_depsNoOp8^gradients_2/current_policy_network/MatMul_1_grad/MatMul:^gradients_2/current_policy_network/MatMul_1_grad/MatMul_1
╨
Igradients_2/current_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity7gradients_2/current_policy_network/MatMul_1_grad/MatMulB^gradients_2/current_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*J
_class@
><loc:@gradients_2/current_policy_network/MatMul_1_grad/MatMul
═
Kgradients_2/current_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity9gradients_2/current_policy_network/MatMul_1_grad/MatMul_1B^gradients_2/current_policy_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:@@*
T0*L
_classB
@>loc:@gradients_2/current_policy_network/MatMul_1_grad/MatMul_1
█
5gradients_2/current_policy_network/Tanh_grad/TanhGradTanhGradcurrent_policy_network/TanhIgradients_2/current_policy_network/MatMul_1_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
╖
Ggradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0current_policy_network/LayerNorm/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
╖
Igradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.current_policy_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeIgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ь
Egradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/SumSum5gradients_2/current_policy_network/Tanh_grad/TanhGradWgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
д
Igradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/SumGgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
а
Ggradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1Sum5gradients_2/current_policy_network/Tanh_grad/TanhGradYgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
Kgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1Igradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*'
_output_shapes
:         @*
T0*
Tshape0
Ї
Rgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
Ц
Zgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @
Ь
\gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:         @
б
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapecurrent_policy_network/add*
_output_shapes
:*
T0*
out_type0
╖
Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.current_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeIgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
К
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMulZgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.current_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
м
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mulWgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/SumGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
°
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1Mulcurrent_policy_network/addZgradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
▓
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1Ygradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
Kgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
Ц
Zgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape
Ь
\gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:         @
П
Egradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:
╖
Ggradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0current_policy_network/LayerNorm/batchnorm/mul_2*
out_type0*
_output_shapes
:*
T0
╗
Ugradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/ShapeGgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┐
Cgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/SumSum\gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Ugradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
Ggradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/SumEgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:@
├
Egradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\gradients_2/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Wgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┤
Cgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/NegNegEgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
в
Igradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/NegGgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
ю
Pgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^gradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^gradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
Б
Xgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^gradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:@*
T0*Z
_classP
NLloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape
Ф
Zgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^gradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @
┤
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-current_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
╖
Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.current_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeIgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
К
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mulMulZgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.current_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
м
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mulWgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/SumGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Л
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-current_policy_network/LayerNorm/moments/meanZgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         @
▓
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Ygradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
к
Kgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpJ^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
Ц
Zgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityIgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeS^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:         
Ь
\gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
ё
gradients_2/AddN_2AddN\gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\gradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:         @
╡
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape0current_policy_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
С
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
╗
Ugradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/ShapeGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╜
Cgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/mulMulgradients_2/AddN_2+current_policy_network/LayerNorm/gamma/read*'
_output_shapes
:         @*
T0
ж
Cgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/SumSumCgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/mulUgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ю
Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/SumEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
─
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0current_policy_network/LayerNorm/batchnorm/Rsqrtgradients_2/AddN_2*
T0*'
_output_shapes
:         @
м
Egradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Wgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
Igradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1Ggradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
ю
Pgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
О
Xgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:         
З
Zgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^gradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:@*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
Ц
Kgradients_2/current_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0current_policy_network/LayerNorm/batchnorm/RsqrtXgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
╢
Egradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape1current_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
К
Ggradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
╗
Ugradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/ShapeGgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
о
Cgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/SumSumKgradients_2/current_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
Ggradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/SumEgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
▓
Egradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumKgradients_2/current_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
У
Igradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeEgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Sum_1Ggradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ю
Pgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpH^gradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/ReshapeJ^gradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
О
Xgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/ReshapeQ^gradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape*'
_output_shapes
:         
Г
Zgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^gradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
┬
Hgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ShapeShape:current_policy_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
ц
Ggradients_2/current_policy_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
▌
Fgradients_2/current_policy_network/LayerNorm/moments/variance_grad/addAddCcurrent_policy_network/LayerNorm/moments/variance/reduction_indicesGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Size*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
х
Fgradients_2/current_policy_network/LayerNorm/moments/variance_grad/modFloorModFgradients_2/current_policy_network/LayerNorm/moments/variance_grad/addGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Size*
_output_shapes
:*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape
ё
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
valueB:*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
э
Ngradients_2/current_policy_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
э
Ngradients_2/current_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
┐
Hgradients_2/current_policy_network/LayerNorm/moments/variance_grad/rangeRangeNgradients_2/current_policy_network/LayerNorm/moments/variance_grad/range/startGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/SizeNgradients_2/current_policy_network/LayerNorm/moments/variance_grad/range/delta*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
ь
Mgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
value	B :*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
ь
Ggradients_2/current_policy_network/LayerNorm/moments/variance_grad/FillFillJgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_1Mgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape
Ъ
Pgradients_2/current_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHgradients_2/current_policy_network/LayerNorm/moments/variance_grad/rangeFgradients_2/current_policy_network/LayerNorm/moments/variance_grad/modHgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ShapeGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Fill*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
N*#
_output_shapes
:         
ы
Lgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
А
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumPgradients_2/current_policy_network/LayerNorm/moments/variance_grad/DynamicStitchLgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:         *
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape
я
Kgradients_2/current_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivHgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ShapeJgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum*
T0*[
_classQ
OMloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
▓
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ReshapeReshapeXgradients_2/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyPgradients_2/current_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
╡
Ggradients_2/current_policy_network/LayerNorm/moments/variance_grad/TileTileJgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ReshapeKgradients_2/current_policy_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
─
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape:current_policy_network/LayerNorm/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
╗
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape1current_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
ё
Hgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
В
Ggradients_2/current_policy_network/LayerNorm/moments/variance_grad/ProdProdJgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2Hgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Const*
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
є
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0
Ж
Igradients_2/current_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdJgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_3Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Const_1*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
я
Ngradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0
Є
Lgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Prod_1Ngradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2
Ё
Mgradients_2/current_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/ProdLgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*]
_classS
QOloc:@gradients_2/current_policy_network/LayerNorm/moments/variance_grad/Shape_2
╬
Ggradients_2/current_policy_network/LayerNorm/moments/variance_grad/CastCastMgradients_2/current_policy_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Щ
Jgradients_2/current_policy_network/LayerNorm/moments/variance_grad/truedivRealDivGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/TileGgradients_2/current_policy_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
л
Qgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapecurrent_policy_network/add*
T0*
out_type0*
_output_shapes
:
╚
Sgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5current_policy_network/LayerNorm/moments/StopGradient*
_output_shapes
:*
T0*
out_type0
▀
agradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeSgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ф
Rgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^gradients_2/current_policy_network/LayerNorm/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
и
Ogradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulRgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarJgradients_2/current_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
и
Ogradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubcurrent_policy_network/add5current_policy_network/LayerNorm/moments/StopGradientK^gradients_2/current_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
м
Qgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mulOgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*'
_output_shapes
:         @*
T0
╠
Ogradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumQgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1agradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┬
Sgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/SumQgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
╨
Qgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1cgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╚
Ugradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1Sgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
▀
Ogradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegUgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:         *
T0
М
\gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpT^gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeP^gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
╛
dgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:         @
╕
fgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg]^gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:         
Ю
Dgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ShapeShapecurrent_policy_network/add*
_output_shapes
:*
T0*
out_type0
▐
Cgradients_2/current_policy_network/LayerNorm/moments/mean_grad/SizeConst*
value	B :*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
═
Bgradients_2/current_policy_network/LayerNorm/moments/mean_grad/addAdd?current_policy_network/LayerNorm/moments/mean/reduction_indicesCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Size*
T0*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
╒
Bgradients_2/current_policy_network/LayerNorm/moments/mean_grad/modFloorModBgradients_2/current_policy_network/LayerNorm/moments/mean_grad/addCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Size*
T0*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
щ
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
valueB:*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
х
Jgradients_2/current_policy_network/LayerNorm/moments/mean_grad/range/startConst*
_output_shapes
: *
value	B : *W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0
х
Jgradients_2/current_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
л
Dgradients_2/current_policy_network/LayerNorm/moments/mean_grad/rangeRangeJgradients_2/current_policy_network/LayerNorm/moments/mean_grad/range/startCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/SizeJgradients_2/current_policy_network/LayerNorm/moments/mean_grad/range/delta*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
ф
Igradients_2/current_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
▄
Cgradients_2/current_policy_network/LayerNorm/moments/mean_grad/FillFillFgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_1Igradients_2/current_policy_network/LayerNorm/moments/mean_grad/Fill/value*
T0*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
В
Lgradients_2/current_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDgradients_2/current_policy_network/LayerNorm/moments/mean_grad/rangeBgradients_2/current_policy_network/LayerNorm/moments/mean_grad/modDgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ShapeCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Fill*#
_output_shapes
:         *
T0*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
N
у
Hgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
Ё
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/MaximumMaximumLgradients_2/current_policy_network/LayerNorm/moments/mean_grad/DynamicStitchHgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*#
_output_shapes
:         
▀
Ggradients_2/current_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivDgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ShapeFgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum*W
_classM
KIloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:*
T0
м
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ReshapeReshapeZgradients_2/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLgradients_2/current_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
й
Cgradients_2/current_policy_network/LayerNorm/moments/mean_grad/TileTileFgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ReshapeGgradients_2/current_policy_network/LayerNorm/moments/mean_grad/floordiv*
T0*0
_output_shapes
:                  *

Tmultiples0
а
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2Shapecurrent_policy_network/add*
T0*
out_type0*
_output_shapes
:
│
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape-current_policy_network/LayerNorm/moments/mean*
T0*
out_type0*
_output_shapes
:
щ
Dgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
Є
Cgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ProdProdFgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2Dgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2
ы
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
Ў
Egradients_2/current_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdFgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_3Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Const_1*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
ч
Jgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
value	B :*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
т
Hgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Prod_1Jgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
р
Igradients_2/current_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/ProdHgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*Y
_classO
MKloc:@gradients_2/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
╞
Cgradients_2/current_policy_network/LayerNorm/moments/mean_grad/CastCastIgradients_2/current_policy_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Н
Fgradients_2/current_policy_network/LayerNorm/moments/mean_grad/truedivRealDivCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/TileCgradients_2/current_policy_network/LayerNorm/moments/mean_grad/Cast*
T0*'
_output_shapes
:         @
╜
gradients_2/AddN_3AddNZgradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydgradients_2/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFgradients_2/current_policy_network/LayerNorm/moments/mean_grad/truediv*
N*'
_output_shapes
:         @*
T0*\
_classR
PNloc:@gradients_2/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape
О
1gradients_2/current_policy_network/add_grad/ShapeShapecurrent_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:
}
3gradients_2/current_policy_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
 
Agradients_2/current_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_2/current_policy_network/add_grad/Shape3gradients_2/current_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
═
/gradients_2/current_policy_network/add_grad/SumSumgradients_2/AddN_3Agradients_2/current_policy_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
т
3gradients_2/current_policy_network/add_grad/ReshapeReshape/gradients_2/current_policy_network/add_grad/Sum1gradients_2/current_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
╤
1gradients_2/current_policy_network/add_grad/Sum_1Sumgradients_2/AddN_3Cgradients_2/current_policy_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
█
5gradients_2/current_policy_network/add_grad/Reshape_1Reshape1gradients_2/current_policy_network/add_grad/Sum_13gradients_2/current_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
▓
<gradients_2/current_policy_network/add_grad/tuple/group_depsNoOp4^gradients_2/current_policy_network/add_grad/Reshape6^gradients_2/current_policy_network/add_grad/Reshape_1
╛
Dgradients_2/current_policy_network/add_grad/tuple/control_dependencyIdentity3gradients_2/current_policy_network/add_grad/Reshape=^gradients_2/current_policy_network/add_grad/tuple/group_deps*F
_class<
:8loc:@gradients_2/current_policy_network/add_grad/Reshape*'
_output_shapes
:         @*
T0
╖
Fgradients_2/current_policy_network/add_grad/tuple/control_dependency_1Identity5gradients_2/current_policy_network/add_grad/Reshape_1=^gradients_2/current_policy_network/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/current_policy_network/add_grad/Reshape_1*
_output_shapes
:@
Ч
5gradients_2/current_policy_network/MatMul_grad/MatMulMatMulDgradients_2/current_policy_network/add_grad/tuple/control_dependency8current_policy_network/current_policy_network/fc0/w/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
ц
7gradients_2/current_policy_network/MatMul_grad/MatMul_1MatMulobservations_2Dgradients_2/current_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
╣
?gradients_2/current_policy_network/MatMul_grad/tuple/group_depsNoOp6^gradients_2/current_policy_network/MatMul_grad/MatMul8^gradients_2/current_policy_network/MatMul_grad/MatMul_1
╚
Ggradients_2/current_policy_network/MatMul_grad/tuple/control_dependencyIdentity5gradients_2/current_policy_network/MatMul_grad/MatMul@^gradients_2/current_policy_network/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/current_policy_network/MatMul_grad/MatMul*'
_output_shapes
:         
┼
Igradients_2/current_policy_network/MatMul_grad/tuple/control_dependency_1Identity7gradients_2/current_policy_network/MatMul_grad/MatMul_1@^gradients_2/current_policy_network/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/current_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:@
Ъ
beta1_power_2/initial_valueConst*
valueB
 *fff?*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
dtype0*
_output_shapes
: 
л
beta1_power_2
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta
╬
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta
И
beta1_power_2/readIdentitybeta1_power_2*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
: 
Ъ
beta2_power_2/initial_valueConst*
_output_shapes
: *
valueB
 *w╛?*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
dtype0
л
beta2_power_2
VariableV2*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
╬
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
_output_shapes
: *
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(
И
beta2_power_2/readIdentitybeta2_power_2*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
: 
ч
Jcurrent_policy_network/current_policy_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB@*    
Ї
8current_policy_network/current_policy_network/fc0/w/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@
щ
?current_policy_network/current_policy_network/fc0/w/Adam/AssignAssign8current_policy_network/current_policy_network/fc0/w/AdamJcurrent_policy_network/current_policy_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w
Ї
=current_policy_network/current_policy_network/fc0/w/Adam/readIdentity8current_policy_network/current_policy_network/fc0/w/Adam*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@*
T0
щ
Lcurrent_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ў
:current_policy_network/current_policy_network/fc0/w/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
	container 
я
Acurrent_policy_network/current_policy_network/fc0/w/Adam_1/AssignAssign:current_policy_network/current_policy_network/fc0/w/Adam_1Lcurrent_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zeros*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
°
?current_policy_network/current_policy_network/fc0/w/Adam_1/readIdentity:current_policy_network/current_policy_network/fc0/w/Adam_1*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
▀
Jcurrent_policy_network/current_policy_network/fc0/b/Adam/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ь
8current_policy_network/current_policy_network/fc0/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
	container *
shape:@
х
?current_policy_network/current_policy_network/fc0/b/Adam/AssignAssign8current_policy_network/current_policy_network/fc0/b/AdamJcurrent_policy_network/current_policy_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
Ё
=current_policy_network/current_policy_network/fc0/b/Adam/readIdentity8current_policy_network/current_policy_network/fc0/b/Adam*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
с
Lcurrent_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ю
:current_policy_network/current_policy_network/fc0/b/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b
ы
Acurrent_policy_network/current_policy_network/fc0/b/Adam_1/AssignAssign:current_policy_network/current_policy_network/fc0/b/Adam_1Lcurrent_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
validate_shape(
Ї
?current_policy_network/current_policy_network/fc0/b/Adam_1/readIdentity:current_policy_network/current_policy_network/fc0/b/Adam_1*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
├
<current_policy_network/LayerNorm/beta/Adam/Initializer/zerosConst*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╨
*current_policy_network/LayerNorm/beta/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape:@
н
1current_policy_network/LayerNorm/beta/Adam/AssignAssign*current_policy_network/LayerNorm/beta/Adam<current_policy_network/LayerNorm/beta/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(
╞
/current_policy_network/LayerNorm/beta/Adam/readIdentity*current_policy_network/LayerNorm/beta/Adam*
_output_shapes
:@*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta
┼
>current_policy_network/LayerNorm/beta/Adam_1/Initializer/zerosConst*
_output_shapes
:@*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
valueB@*    *
dtype0
╥
,current_policy_network/LayerNorm/beta/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container 
│
3current_policy_network/LayerNorm/beta/Adam_1/AssignAssign,current_policy_network/LayerNorm/beta/Adam_1>current_policy_network/LayerNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
╩
1current_policy_network/LayerNorm/beta/Adam_1/readIdentity,current_policy_network/LayerNorm/beta/Adam_1*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
:@
┼
=current_policy_network/LayerNorm/gamma/Adam/Initializer/zerosConst*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╥
+current_policy_network/LayerNorm/gamma/Adam
VariableV2*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
▒
2current_policy_network/LayerNorm/gamma/Adam/AssignAssign+current_policy_network/LayerNorm/gamma/Adam=current_policy_network/LayerNorm/gamma/Adam/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
╔
0current_policy_network/LayerNorm/gamma/Adam/readIdentity+current_policy_network/LayerNorm/gamma/Adam*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
_output_shapes
:@
╟
?current_policy_network/LayerNorm/gamma/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╘
-current_policy_network/LayerNorm/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
	container *
shape:@
╖
4current_policy_network/LayerNorm/gamma/Adam_1/AssignAssign-current_policy_network/LayerNorm/gamma/Adam_1?current_policy_network/LayerNorm/gamma/Adam_1/Initializer/zeros*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
═
2current_policy_network/LayerNorm/gamma/Adam_1/readIdentity-current_policy_network/LayerNorm/gamma/Adam_1*
_output_shapes
:@*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma
ч
Jcurrent_policy_network/current_policy_network/fc1/w/Adam/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
Ї
8current_policy_network/current_policy_network/fc1/w/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@
щ
?current_policy_network/current_policy_network/fc1/w/Adam/AssignAssign8current_policy_network/current_policy_network/fc1/w/AdamJcurrent_policy_network/current_policy_network/fc1/w/Adam/Initializer/zeros*
_output_shapes

:@@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
validate_shape(
Ї
=current_policy_network/current_policy_network/fc1/w/Adam/readIdentity8current_policy_network/current_policy_network/fc1/w/Adam*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@*
T0
щ
Lcurrent_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0
Ў
:current_policy_network/current_policy_network/fc1/w/Adam_1
VariableV2*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
я
Acurrent_policy_network/current_policy_network/fc1/w/Adam_1/AssignAssign:current_policy_network/current_policy_network/fc1/w/Adam_1Lcurrent_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
°
?current_policy_network/current_policy_network/fc1/w/Adam_1/readIdentity:current_policy_network/current_policy_network/fc1/w/Adam_1*
_output_shapes

:@@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
▀
Jcurrent_policy_network/current_policy_network/fc1/b/Adam/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ь
8current_policy_network/current_policy_network/fc1/b/Adam
VariableV2*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
х
?current_policy_network/current_policy_network/fc1/b/Adam/AssignAssign8current_policy_network/current_policy_network/fc1/b/AdamJcurrent_policy_network/current_policy_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
Ё
=current_policy_network/current_policy_network/fc1/b/Adam/readIdentity8current_policy_network/current_policy_network/fc1/b/Adam*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@*
T0
с
Lcurrent_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ю
:current_policy_network/current_policy_network/fc1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
	container *
shape:@
ы
Acurrent_policy_network/current_policy_network/fc1/b/Adam_1/AssignAssign:current_policy_network/current_policy_network/fc1/b/Adam_1Lcurrent_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
Ї
?current_policy_network/current_policy_network/fc1/b/Adam_1/readIdentity:current_policy_network/current_policy_network/fc1/b/Adam_1*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@*
T0
╟
>current_policy_network/LayerNorm_1/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
valueB@*    
╘
,current_policy_network/LayerNorm_1/beta/Adam
VariableV2*
shared_name *:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
╡
3current_policy_network/LayerNorm_1/beta/Adam/AssignAssign,current_policy_network/LayerNorm_1/beta/Adam>current_policy_network/LayerNorm_1/beta/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
validate_shape(
╠
1current_policy_network/LayerNorm_1/beta/Adam/readIdentity,current_policy_network/LayerNorm_1/beta/Adam*
_output_shapes
:@*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta
╔
@current_policy_network/LayerNorm_1/beta/Adam_1/Initializer/zerosConst*
_output_shapes
:@*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
valueB@*    *
dtype0
╓
.current_policy_network/LayerNorm_1/beta/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta
╗
5current_policy_network/LayerNorm_1/beta/Adam_1/AssignAssign.current_policy_network/LayerNorm_1/beta/Adam_1@current_policy_network/LayerNorm_1/beta/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
validate_shape(
╨
3current_policy_network/LayerNorm_1/beta/Adam_1/readIdentity.current_policy_network/LayerNorm_1/beta/Adam_1*
_output_shapes
:@*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta
╔
?current_policy_network/LayerNorm_1/gamma/Adam/Initializer/zerosConst*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╓
-current_policy_network/LayerNorm_1/gamma/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
	container *
shape:@
╣
4current_policy_network/LayerNorm_1/gamma/Adam/AssignAssign-current_policy_network/LayerNorm_1/gamma/Adam?current_policy_network/LayerNorm_1/gamma/Adam/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╧
2current_policy_network/LayerNorm_1/gamma/Adam/readIdentity-current_policy_network/LayerNorm_1/gamma/Adam*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
_output_shapes
:@
╦
Acurrent_policy_network/LayerNorm_1/gamma/Adam_1/Initializer/zerosConst*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╪
/current_policy_network/LayerNorm_1/gamma/Adam_1
VariableV2*
shared_name *;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
┐
6current_policy_network/LayerNorm_1/gamma/Adam_1/AssignAssign/current_policy_network/LayerNorm_1/gamma/Adam_1Acurrent_policy_network/LayerNorm_1/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╙
4current_policy_network/LayerNorm_1/gamma/Adam_1/readIdentity/current_policy_network/LayerNorm_1/gamma/Adam_1*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
_output_shapes
:@
ч
Jcurrent_policy_network/current_policy_network/out/w/Adam/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ї
8current_policy_network/current_policy_network/out/w/Adam
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/w
щ
?current_policy_network/current_policy_network/out/w/Adam/AssignAssign8current_policy_network/current_policy_network/out/w/AdamJcurrent_policy_network/current_policy_network/out/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w
Ї
=current_policy_network/current_policy_network/out/w/Adam/readIdentity8current_policy_network/current_policy_network/out/w/Adam*
_output_shapes

:@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w
щ
Lcurrent_policy_network/current_policy_network/out/w/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ў
:current_policy_network/current_policy_network/out/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
	container *
shape
:@
я
Acurrent_policy_network/current_policy_network/out/w/Adam_1/AssignAssign:current_policy_network/current_policy_network/out/w/Adam_1Lcurrent_policy_network/current_policy_network/out/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w
°
?current_policy_network/current_policy_network/out/w/Adam_1/readIdentity:current_policy_network/current_policy_network/out/w/Adam_1*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
_output_shapes

:@
▀
Jcurrent_policy_network/current_policy_network/out/b/Adam/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ь
8current_policy_network/current_policy_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
	container *
shape:
х
?current_policy_network/current_policy_network/out/b/Adam/AssignAssign8current_policy_network/current_policy_network/out/b/AdamJcurrent_policy_network/current_policy_network/out/b/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
Ё
=current_policy_network/current_policy_network/out/b/Adam/readIdentity8current_policy_network/current_policy_network/out/b/Adam*
_output_shapes
:*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b
с
Lcurrent_policy_network/current_policy_network/out/b/Adam_1/Initializer/zerosConst*
_output_shapes
:*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0
ю
:current_policy_network/current_policy_network/out/b/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
ы
Acurrent_policy_network/current_policy_network/out/b/Adam_1/AssignAssign:current_policy_network/current_policy_network/out/b/Adam_1Lcurrent_policy_network/current_policy_network/out/b/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
Ї
?current_policy_network/current_policy_network/out/b/Adam_1/readIdentity:current_policy_network/current_policy_network/out/b/Adam_1*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
_output_shapes
:
Q
Adam_2/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_2/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w╛?
S
Adam_2/epsilonConst*
_output_shapes
: *
valueB
 *w╠+2*
dtype0
╦
KAdam_2/update_current_policy_network/current_policy_network/fc0/w/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc0/w8current_policy_network/current_policy_network/fc0/w/Adam:current_policy_network/current_policy_network/fc0/w/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonIgradients_2/current_policy_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:@
─
KAdam_2/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc0/b8current_policy_network/current_policy_network/fc0/b/Adam:current_policy_network/current_policy_network/fc0/b/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonFgradients_2/current_policy_network/add_grad/tuple/control_dependency_1*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
Р
=Adam_2/update_current_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam%current_policy_network/LayerNorm/beta*current_policy_network/LayerNorm/beta/Adam,current_policy_network/LayerNorm/beta/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonXgradients_2/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
Ч
>Adam_2/update_current_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&current_policy_network/LayerNorm/gamma+current_policy_network/LayerNorm/gamma/Adam-current_policy_network/LayerNorm/gamma/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonZgradients_2/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
use_nesterov( *
_output_shapes
:@
═
KAdam_2/update_current_policy_network/current_policy_network/fc1/w/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc1/w8current_policy_network/current_policy_network/fc1/w/Adam:current_policy_network/current_policy_network/fc1/w/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonKgradients_2/current_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@@*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
╞
KAdam_2/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc1/b8current_policy_network/current_policy_network/fc1/b/Adam:current_policy_network/current_policy_network/fc1/b/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonHgradients_2/current_policy_network/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
use_nesterov( *
_output_shapes
:@
Ь
?Adam_2/update_current_policy_network/LayerNorm_1/beta/ApplyAdam	ApplyAdam'current_policy_network/LayerNorm_1/beta,current_policy_network/LayerNorm_1/beta/Adam.current_policy_network/LayerNorm_1/beta/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonZgradients_2/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
use_nesterov( *
_output_shapes
:@
г
@Adam_2/update_current_policy_network/LayerNorm_1/gamma/ApplyAdam	ApplyAdam(current_policy_network/LayerNorm_1/gamma-current_policy_network/LayerNorm_1/gamma/Adam/current_policy_network/LayerNorm_1/gamma/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon\gradients_2/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
use_nesterov( *
_output_shapes
:@
═
KAdam_2/update_current_policy_network/current_policy_network/out/w/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/out/w8current_policy_network/current_policy_network/out/w/Adam:current_policy_network/current_policy_network/out/w/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonKgradients_2/current_policy_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
use_nesterov( *
_output_shapes

:@
╞
KAdam_2/update_current_policy_network/current_policy_network/out/b/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/out/b8current_policy_network/current_policy_network/out/b/Adam:current_policy_network/current_policy_network/out/b/Adam_1beta1_power_2/readbeta2_power_2/readlearning_rate_2Adam_2/beta1Adam_2/beta2Adam_2/epsilonHgradients_2/current_policy_network/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
use_nesterov( 
ш

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1L^Adam_2/update_current_policy_network/current_policy_network/fc0/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam>^Adam_2/update_current_policy_network/LayerNorm/beta/ApplyAdam?^Adam_2/update_current_policy_network/LayerNorm/gamma/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc1/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam@^Adam_2/update_current_policy_network/LayerNorm_1/beta/ApplyAdamA^Adam_2/update_current_policy_network/LayerNorm_1/gamma/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/out/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta
╢
Adam_2/AssignAssignbeta1_power_2
Adam_2/mul*
use_locking( *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
ъ
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2L^Adam_2/update_current_policy_network/current_policy_network/fc0/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam>^Adam_2/update_current_policy_network/LayerNorm/beta/ApplyAdam?^Adam_2/update_current_policy_network/LayerNorm/gamma/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc1/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam@^Adam_2/update_current_policy_network/LayerNorm_1/beta/ApplyAdamA^Adam_2/update_current_policy_network/LayerNorm_1/gamma/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/out/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/out/b/ApplyAdam*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
: 
║
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
use_locking( *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
К
Adam_2NoOpL^Adam_2/update_current_policy_network/current_policy_network/fc0/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam>^Adam_2/update_current_policy_network/LayerNorm/beta/ApplyAdam?^Adam_2/update_current_policy_network/LayerNorm/gamma/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc1/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam@^Adam_2/update_current_policy_network/LayerNorm_1/beta/ApplyAdamA^Adam_2/update_current_policy_network/LayerNorm_1/gamma/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/out/w/ApplyAdamL^Adam_2/update_current_policy_network/current_policy_network/out/b/ApplyAdam^Adam_2/Assign^Adam_2/Assign_1
q
!Normal_7/log_prob/standardize/subSub	actions_1
Normal/loc*
T0*'
_output_shapes
:         
Д
%Normal_7/log_prob/standardize/truedivRealDiv!Normal_7/log_prob/standardize/subNormal/scale*
T0*
_output_shapes
:
l
Normal_7/log_prob/SquareSquare%Normal_7/log_prob/standardize/truediv*
T0*
_output_shapes
:
\
Normal_7/log_prob/mul/xConst*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
r
Normal_7/log_prob/mulMulNormal_7/log_prob/mul/xNormal_7/log_prob/Square*
T0*
_output_shapes
:
M
Normal_7/log_prob/LogLogNormal/scale*
T0*
_output_shapes
:
\
Normal_7/log_prob/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *О?k?
o
Normal_7/log_prob/addAddNormal_7/log_prob/add/xNormal_7/log_prob/Log*
T0*
_output_shapes
:
m
Normal_7/log_prob/subSubNormal_7/log_prob/mulNormal_7/log_prob/add*
T0*
_output_shapes
:
D
NegNegNormal_7/log_prob/sub*
_output_shapes
:*
T0
@
mul_3MulNeg
advantages*
T0*
_output_shapes
:
6
Rank_1Rankmul_3*
_output_shapes
: *
T0
O
range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
g
range_1Rangerange_1/startRank_1range_1/delta*#
_output_shapes
:         *

Tidx0
^
Mean_5Meanmul_3range_1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
l
policy_network_loss/tagsConst*$
valueB Bpolicy_network_loss*
dtype0*
_output_shapes
: 
g
policy_network_lossScalarSummarypolicy_network_loss/tagsMean_5*
T0*
_output_shapes
: 
`
gradients_3/ShapeShapeMean_5*#
_output_shapes
:         *
T0*
out_type0
V
gradients_3/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
a
gradients_3/FillFillgradients_3/Shapegradients_3/Const*
_output_shapes
:*
T0
k
gradients_3/Mean_5_grad/ShapeShapemul_3*
out_type0*#
_output_shapes
:         *
T0
ж
gradients_3/Mean_5_grad/SizeSizegradients_3/Mean_5_grad/Shape*
_output_shapes
: *
T0*
out_type0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape
й
gradients_3/Mean_5_grad/addAddrange_1gradients_3/Mean_5_grad/Size*
T0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*#
_output_shapes
:         
┬
gradients_3/Mean_5_grad/modFloorModgradients_3/Mean_5_grad/addgradients_3/Mean_5_grad/Size*#
_output_shapes
:         *
T0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape
м
gradients_3/Mean_5_grad/Shape_1Shapegradients_3/Mean_5_grad/mod*
T0*
out_type0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*
_output_shapes
:
Ч
#gradients_3/Mean_5_grad/range/startConst*
value	B : *0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*
dtype0*
_output_shapes
: 
Ч
#gradients_3/Mean_5_grad/range/deltaConst*
value	B :*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*
dtype0*
_output_shapes
: 
ё
gradients_3/Mean_5_grad/rangeRange#gradients_3/Mean_5_grad/range/startgradients_3/Mean_5_grad/Size#gradients_3/Mean_5_grad/range/delta*#
_output_shapes
:         *

Tidx0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape
Ц
"gradients_3/Mean_5_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape
╔
gradients_3/Mean_5_grad/FillFillgradients_3/Mean_5_grad/Shape_1"gradients_3/Mean_5_grad/Fill/value*#
_output_shapes
:         *
T0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape
Ш
%gradients_3/Mean_5_grad/DynamicStitchDynamicStitchgradients_3/Mean_5_grad/rangegradients_3/Mean_5_grad/modgradients_3/Mean_5_grad/Shapegradients_3/Mean_5_grad/Fill*
T0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*
N*#
_output_shapes
:         
Х
!gradients_3/Mean_5_grad/Maximum/yConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*
dtype0
╘
gradients_3/Mean_5_grad/MaximumMaximum%gradients_3/Mean_5_grad/DynamicStitch!gradients_3/Mean_5_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*#
_output_shapes
:         
╠
 gradients_3/Mean_5_grad/floordivFloorDivgradients_3/Mean_5_grad/Shapegradients_3/Mean_5_grad/Maximum*
T0*0
_class&
$"loc:@gradients_3/Mean_5_grad/Shape*#
_output_shapes
:         
Ф
gradients_3/Mean_5_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ь
gradients_3/Mean_5_grad/TileTilegradients_3/Mean_5_grad/Reshape gradients_3/Mean_5_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
m
gradients_3/Mean_5_grad/Shape_2Shapemul_3*
T0*
out_type0*#
_output_shapes
:         
n
gradients_3/Mean_5_grad/Shape_3ShapeMean_5*
T0*
out_type0*#
_output_shapes
:         
Ы
gradients_3/Mean_5_grad/ConstConst*
valueB: *2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2*
dtype0*
_output_shapes
:
╓
gradients_3/Mean_5_grad/ProdProdgradients_3/Mean_5_grad/Shape_2gradients_3/Mean_5_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2
Э
gradients_3/Mean_5_grad/Const_1Const*
valueB: *2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2*
dtype0*
_output_shapes
:
┌
gradients_3/Mean_5_grad/Prod_1Prodgradients_3/Mean_5_grad/Shape_3gradients_3/Mean_5_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2
Щ
#gradients_3/Mean_5_grad/Maximum_1/yConst*
value	B :*2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2*
dtype0*
_output_shapes
: 
╞
!gradients_3/Mean_5_grad/Maximum_1Maximumgradients_3/Mean_5_grad/Prod_1#gradients_3/Mean_5_grad/Maximum_1/y*
T0*2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2*
_output_shapes
: 
─
"gradients_3/Mean_5_grad/floordiv_1FloorDivgradients_3/Mean_5_grad/Prod!gradients_3/Mean_5_grad/Maximum_1*
T0*2
_class(
&$loc:@gradients_3/Mean_5_grad/Shape_2*
_output_shapes
: 
x
gradients_3/Mean_5_grad/CastCast"gradients_3/Mean_5_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Й
gradients_3/Mean_5_grad/truedivRealDivgradients_3/Mean_5_grad/Tilegradients_3/Mean_5_grad/Cast*
_output_shapes
:*
T0
h
gradients_3/mul_3_grad/ShapeShapeNeg*#
_output_shapes
:         *
T0*
out_type0
h
gradients_3/mul_3_grad/Shape_1Shape
advantages*
out_type0*
_output_shapes
:*
T0
└
,gradients_3/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_3_grad/Shapegradients_3/mul_3_grad/Shape_1*2
_output_shapes 
:         :         *
T0
q
gradients_3/mul_3_grad/mulMulgradients_3/Mean_5_grad/truediv
advantages*
T0*
_output_shapes
:
л
gradients_3/mul_3_grad/SumSumgradients_3/mul_3_grad/mul,gradients_3/mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ф
gradients_3/mul_3_grad/ReshapeReshapegradients_3/mul_3_grad/Sumgradients_3/mul_3_grad/Shape*
T0*
Tshape0*
_output_shapes
:
l
gradients_3/mul_3_grad/mul_1MulNeggradients_3/Mean_5_grad/truediv*
_output_shapes
:*
T0
▒
gradients_3/mul_3_grad/Sum_1Sumgradients_3/mul_3_grad/mul_1.gradients_3/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
й
 gradients_3/mul_3_grad/Reshape_1Reshapegradients_3/mul_3_grad/Sum_1gradients_3/mul_3_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
s
'gradients_3/mul_3_grad/tuple/group_depsNoOp^gradients_3/mul_3_grad/Reshape!^gradients_3/mul_3_grad/Reshape_1
█
/gradients_3/mul_3_grad/tuple/control_dependencyIdentitygradients_3/mul_3_grad/Reshape(^gradients_3/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_3/mul_3_grad/Reshape*
_output_shapes
:
Ё
1gradients_3/mul_3_grad/tuple/control_dependency_1Identity gradients_3/mul_3_grad/Reshape_1(^gradients_3/mul_3_grad/tuple/group_deps*'
_output_shapes
:         *
T0*3
_class)
'%loc:@gradients_3/mul_3_grad/Reshape_1
s
gradients_3/Neg_grad/NegNeg/gradients_3/mul_3_grad/tuple/control_dependency*
T0*
_output_shapes
:
К
,gradients_3/Normal_7/log_prob/sub_grad/ShapeShapeNormal_7/log_prob/mul*
T0*
out_type0*#
_output_shapes
:         
М
.gradients_3/Normal_7/log_prob/sub_grad/Shape_1ShapeNormal_7/log_prob/add*#
_output_shapes
:         *
T0*
out_type0
Ё
<gradients_3/Normal_7/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_3/Normal_7/log_prob/sub_grad/Shape.gradients_3/Normal_7/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╔
*gradients_3/Normal_7/log_prob/sub_grad/SumSumgradients_3/Neg_grad/Neg<gradients_3/Normal_7/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
─
.gradients_3/Normal_7/log_prob/sub_grad/ReshapeReshape*gradients_3/Normal_7/log_prob/sub_grad/Sum,gradients_3/Normal_7/log_prob/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
═
,gradients_3/Normal_7/log_prob/sub_grad/Sum_1Sumgradients_3/Neg_grad/Neg>gradients_3/Normal_7/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
В
*gradients_3/Normal_7/log_prob/sub_grad/NegNeg,gradients_3/Normal_7/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
╚
0gradients_3/Normal_7/log_prob/sub_grad/Reshape_1Reshape*gradients_3/Normal_7/log_prob/sub_grad/Neg.gradients_3/Normal_7/log_prob/sub_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
г
7gradients_3/Normal_7/log_prob/sub_grad/tuple/group_depsNoOp/^gradients_3/Normal_7/log_prob/sub_grad/Reshape1^gradients_3/Normal_7/log_prob/sub_grad/Reshape_1
Ы
?gradients_3/Normal_7/log_prob/sub_grad/tuple/control_dependencyIdentity.gradients_3/Normal_7/log_prob/sub_grad/Reshape8^gradients_3/Normal_7/log_prob/sub_grad/tuple/group_deps*A
_class7
53loc:@gradients_3/Normal_7/log_prob/sub_grad/Reshape*
_output_shapes
:*
T0
б
Agradients_3/Normal_7/log_prob/sub_grad/tuple/control_dependency_1Identity0gradients_3/Normal_7/log_prob/sub_grad/Reshape_18^gradients_3/Normal_7/log_prob/sub_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_3/Normal_7/log_prob/sub_grad/Reshape_1*
_output_shapes
:
o
,gradients_3/Normal_7/log_prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
П
.gradients_3/Normal_7/log_prob/mul_grad/Shape_1ShapeNormal_7/log_prob/Square*#
_output_shapes
:         *
T0*
out_type0
Ё
<gradients_3/Normal_7/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_3/Normal_7/log_prob/mul_grad/Shape.gradients_3/Normal_7/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
п
*gradients_3/Normal_7/log_prob/mul_grad/mulMul?gradients_3/Normal_7/log_prob/sub_grad/tuple/control_dependencyNormal_7/log_prob/Square*
T0*
_output_shapes
:
█
*gradients_3/Normal_7/log_prob/mul_grad/SumSum*gradients_3/Normal_7/log_prob/mul_grad/mul<gradients_3/Normal_7/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┬
.gradients_3/Normal_7/log_prob/mul_grad/ReshapeReshape*gradients_3/Normal_7/log_prob/mul_grad/Sum,gradients_3/Normal_7/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
░
,gradients_3/Normal_7/log_prob/mul_grad/mul_1MulNormal_7/log_prob/mul/x?gradients_3/Normal_7/log_prob/sub_grad/tuple/control_dependency*
_output_shapes
:*
T0
с
,gradients_3/Normal_7/log_prob/mul_grad/Sum_1Sum,gradients_3/Normal_7/log_prob/mul_grad/mul_1>gradients_3/Normal_7/log_prob/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╩
0gradients_3/Normal_7/log_prob/mul_grad/Reshape_1Reshape,gradients_3/Normal_7/log_prob/mul_grad/Sum_1.gradients_3/Normal_7/log_prob/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
г
7gradients_3/Normal_7/log_prob/mul_grad/tuple/group_depsNoOp/^gradients_3/Normal_7/log_prob/mul_grad/Reshape1^gradients_3/Normal_7/log_prob/mul_grad/Reshape_1
Щ
?gradients_3/Normal_7/log_prob/mul_grad/tuple/control_dependencyIdentity.gradients_3/Normal_7/log_prob/mul_grad/Reshape8^gradients_3/Normal_7/log_prob/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_3/Normal_7/log_prob/mul_grad/Reshape*
_output_shapes
: 
б
Agradients_3/Normal_7/log_prob/mul_grad/tuple/control_dependency_1Identity0gradients_3/Normal_7/log_prob/mul_grad/Reshape_18^gradients_3/Normal_7/log_prob/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*C
_class9
75loc:@gradients_3/Normal_7/log_prob/mul_grad/Reshape_1
╕
/gradients_3/Normal_7/log_prob/Square_grad/mul/xConstB^gradients_3/Normal_7/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
п
-gradients_3/Normal_7/log_prob/Square_grad/mulMul/gradients_3/Normal_7/log_prob/Square_grad/mul/x%Normal_7/log_prob/standardize/truediv*
T0*
_output_shapes
:
╦
/gradients_3/Normal_7/log_prob/Square_grad/mul_1MulAgradients_3/Normal_7/log_prob/mul_grad/tuple/control_dependency_1-gradients_3/Normal_7/log_prob/Square_grad/mul*
_output_shapes
:*
T0
Э
<gradients_3/Normal_7/log_prob/standardize/truediv_grad/ShapeShape!Normal_7/log_prob/standardize/sub*
T0*
out_type0*
_output_shapes
:
У
>gradients_3/Normal_7/log_prob/standardize/truediv_grad/Shape_1ShapeNormal/scale*#
_output_shapes
:         *
T0*
out_type0
а
Lgradients_3/Normal_7/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_3/Normal_7/log_prob/standardize/truediv_grad/Shape>gradients_3/Normal_7/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:         :         *
T0
л
>gradients_3/Normal_7/log_prob/standardize/truediv_grad/RealDivRealDiv/gradients_3/Normal_7/log_prob/Square_grad/mul_1Normal/scale*
T0*
_output_shapes
:
П
:gradients_3/Normal_7/log_prob/standardize/truediv_grad/SumSum>gradients_3/Normal_7/log_prob/standardize/truediv_grad/RealDivLgradients_3/Normal_7/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
>gradients_3/Normal_7/log_prob/standardize/truediv_grad/ReshapeReshape:gradients_3/Normal_7/log_prob/standardize/truediv_grad/Sum<gradients_3/Normal_7/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ц
:gradients_3/Normal_7/log_prob/standardize/truediv_grad/NegNeg!Normal_7/log_prob/standardize/sub*
T0*'
_output_shapes
:         
╕
@gradients_3/Normal_7/log_prob/standardize/truediv_grad/RealDiv_1RealDiv:gradients_3/Normal_7/log_prob/standardize/truediv_grad/NegNormal/scale*
_output_shapes
:*
T0
╛
@gradients_3/Normal_7/log_prob/standardize/truediv_grad/RealDiv_2RealDiv@gradients_3/Normal_7/log_prob/standardize/truediv_grad/RealDiv_1Normal/scale*
T0*
_output_shapes
:
╫
:gradients_3/Normal_7/log_prob/standardize/truediv_grad/mulMul/gradients_3/Normal_7/log_prob/Square_grad/mul_1@gradients_3/Normal_7/log_prob/standardize/truediv_grad/RealDiv_2*
_output_shapes
:*
T0
П
<gradients_3/Normal_7/log_prob/standardize/truediv_grad/Sum_1Sum:gradients_3/Normal_7/log_prob/standardize/truediv_grad/mulNgradients_3/Normal_7/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
·
@gradients_3/Normal_7/log_prob/standardize/truediv_grad/Reshape_1Reshape<gradients_3/Normal_7/log_prob/standardize/truediv_grad/Sum_1>gradients_3/Normal_7/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
╙
Ggradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/group_depsNoOp?^gradients_3/Normal_7/log_prob/standardize/truediv_grad/ReshapeA^gradients_3/Normal_7/log_prob/standardize/truediv_grad/Reshape_1
ъ
Ogradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentity>gradients_3/Normal_7/log_prob/standardize/truediv_grad/ReshapeH^gradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/group_deps*Q
_classG
ECloc:@gradients_3/Normal_7/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:         *
T0
с
Qgradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/control_dependency_1Identity@gradients_3/Normal_7/log_prob/standardize/truediv_grad/Reshape_1H^gradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_3/Normal_7/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
:
Б
8gradients_3/Normal_7/log_prob/standardize/sub_grad/ShapeShape	actions_1*
T0*
out_type0*
_output_shapes
:
Д
:gradients_3/Normal_7/log_prob/standardize/sub_grad/Shape_1Shape
Normal/loc*
T0*
out_type0*
_output_shapes
:
Ф
Hgradients_3/Normal_7/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_3/Normal_7/log_prob/standardize/sub_grad/Shape:gradients_3/Normal_7/log_prob/standardize/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ш
6gradients_3/Normal_7/log_prob/standardize/sub_grad/SumSumOgradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/control_dependencyHgradients_3/Normal_7/log_prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
:gradients_3/Normal_7/log_prob/standardize/sub_grad/ReshapeReshape6gradients_3/Normal_7/log_prob/standardize/sub_grad/Sum8gradients_3/Normal_7/log_prob/standardize/sub_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Ь
8gradients_3/Normal_7/log_prob/standardize/sub_grad/Sum_1SumOgradients_3/Normal_7/log_prob/standardize/truediv_grad/tuple/control_dependencyJgradients_3/Normal_7/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ъ
6gradients_3/Normal_7/log_prob/standardize/sub_grad/NegNeg8gradients_3/Normal_7/log_prob/standardize/sub_grad/Sum_1*
_output_shapes
:*
T0
√
<gradients_3/Normal_7/log_prob/standardize/sub_grad/Reshape_1Reshape6gradients_3/Normal_7/log_prob/standardize/sub_grad/Neg:gradients_3/Normal_7/log_prob/standardize/sub_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
╟
Cgradients_3/Normal_7/log_prob/standardize/sub_grad/tuple/group_depsNoOp;^gradients_3/Normal_7/log_prob/standardize/sub_grad/Reshape=^gradients_3/Normal_7/log_prob/standardize/sub_grad/Reshape_1
┌
Kgradients_3/Normal_7/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity:gradients_3/Normal_7/log_prob/standardize/sub_grad/ReshapeD^gradients_3/Normal_7/log_prob/standardize/sub_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_3/Normal_7/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:         
р
Mgradients_3/Normal_7/log_prob/standardize/sub_grad/tuple/control_dependency_1Identity<gradients_3/Normal_7/log_prob/standardize/sub_grad/Reshape_1D^gradients_3/Normal_7/log_prob/standardize/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_3/Normal_7/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:         
n
gradients_3/Reshape_grad/ShapeShapeSqueeze*
T0*
out_type0*#
_output_shapes
:         
╦
 gradients_3/Reshape_grad/ReshapeReshapeMgradients_3/Normal_7/log_prob/standardize/sub_grad/tuple/control_dependency_1gradients_3/Reshape_grad/Shape*
_output_shapes
:*
T0*
Tshape0
k
gradients_3/Squeeze_grad/ShapeShapestrided_slice*
T0*
out_type0*
_output_shapes
:
н
 gradients_3/Squeeze_grad/ReshapeReshape gradients_3/Reshape_grad/Reshapegradients_3/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
А
$gradients_3/strided_slice_grad/ShapeShapecurrent_policy_network/add_2*
T0*
out_type0*
_output_shapes
:
ё
/gradients_3/strided_slice_grad/StridedSliceGradStridedSliceGrad$gradients_3/strided_slice_grad/Shapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2 gradients_3/Squeeze_grad/Reshape*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:         *
T0*
Index0*
shrink_axis_mask 
Т
3gradients_3/current_policy_network/add_2_grad/ShapeShapecurrent_policy_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

5gradients_3/current_policy_network/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
Е
Cgradients_3/current_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_3/current_policy_network/add_2_grad/Shape5gradients_3/current_policy_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ю
1gradients_3/current_policy_network/add_2_grad/SumSum/gradients_3/strided_slice_grad/StridedSliceGradCgradients_3/current_policy_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ш
5gradients_3/current_policy_network/add_2_grad/ReshapeReshape1gradients_3/current_policy_network/add_2_grad/Sum3gradients_3/current_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Є
3gradients_3/current_policy_network/add_2_grad/Sum_1Sum/gradients_3/strided_slice_grad/StridedSliceGradEgradients_3/current_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
с
7gradients_3/current_policy_network/add_2_grad/Reshape_1Reshape3gradients_3/current_policy_network/add_2_grad/Sum_15gradients_3/current_policy_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
╕
>gradients_3/current_policy_network/add_2_grad/tuple/group_depsNoOp6^gradients_3/current_policy_network/add_2_grad/Reshape8^gradients_3/current_policy_network/add_2_grad/Reshape_1
╞
Fgradients_3/current_policy_network/add_2_grad/tuple/control_dependencyIdentity5gradients_3/current_policy_network/add_2_grad/Reshape?^gradients_3/current_policy_network/add_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/current_policy_network/add_2_grad/Reshape*'
_output_shapes
:         
┐
Hgradients_3/current_policy_network/add_2_grad/tuple/control_dependency_1Identity7gradients_3/current_policy_network/add_2_grad/Reshape_1?^gradients_3/current_policy_network/add_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_3/current_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
Ы
7gradients_3/current_policy_network/MatMul_2_grad/MatMulMatMulFgradients_3/current_policy_network/add_2_grad/tuple/control_dependency8current_policy_network/current_policy_network/out/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
∙
9gradients_3/current_policy_network/MatMul_2_grad/MatMul_1MatMulcurrent_policy_network/Tanh_1Fgradients_3/current_policy_network/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
┐
Agradients_3/current_policy_network/MatMul_2_grad/tuple/group_depsNoOp8^gradients_3/current_policy_network/MatMul_2_grad/MatMul:^gradients_3/current_policy_network/MatMul_2_grad/MatMul_1
╨
Igradients_3/current_policy_network/MatMul_2_grad/tuple/control_dependencyIdentity7gradients_3/current_policy_network/MatMul_2_grad/MatMulB^gradients_3/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_3/current_policy_network/MatMul_2_grad/MatMul*'
_output_shapes
:         @
═
Kgradients_3/current_policy_network/MatMul_2_grad/tuple/control_dependency_1Identity9gradients_3/current_policy_network/MatMul_2_grad/MatMul_1B^gradients_3/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_3/current_policy_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
▀
7gradients_3/current_policy_network/Tanh_1_grad/TanhGradTanhGradcurrent_policy_network/Tanh_1Igradients_3/current_policy_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
╗
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ShapeShape2current_policy_network/LayerNorm_1/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
╗
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1Shape0current_policy_network/LayerNorm_1/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
╟
Ygradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ShapeKgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
в
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/SumSum7gradients_3/current_policy_network/Tanh_1_grad/TanhGradYgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
к
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeReshapeGgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/SumIgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
ж
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Sum7gradients_3/current_policy_network/Tanh_1_grad/TanhGrad[gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
░
Mgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1ReshapeIgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Sum_1Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
·
Tgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_depsNoOpL^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeN^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1
Ю
\gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependencyIdentityKgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/ReshapeU^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @*
T0
д
^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1IdentityMgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1U^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:         @
е
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeShapecurrent_policy_network/add_1*
out_type0*
_output_shapes
:*
T0
╗
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1Shape0current_policy_network/LayerNorm_1/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
╟
Ygradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ShapeKgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mulMul\gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency0current_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
▓
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/SumSumGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mulYgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeReshapeGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/SumIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
■
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1Mulcurrent_policy_network/add_1\gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
╕
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1SumIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/mul_1[gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
░
Mgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1ReshapeIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Sum_1Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Shape_1*
Tshape0*'
_output_shapes
:         @*
T0
·
Tgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_depsNoOpL^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeN^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1
Ю
\gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyIdentityKgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/ReshapeU^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:         @
д
^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityMgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1U^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*`
_classV
TRloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1
С
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:
╗
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape_1Shape2current_policy_network/LayerNorm_1/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ShapeIgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┼
Egradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/SumSum^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Wgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ReshapeReshapeEgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/SumGgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:@
╔
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Sum_1Sum^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_1_grad/tuple/control_dependency_1Ygradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╕
Egradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/NegNegGgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
и
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1ReshapeEgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/NegIgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_depsNoOpJ^gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ReshapeL^gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1
Й
Zgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependencyIdentityIgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/ReshapeS^gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape*
_output_shapes
:@
Ь
\gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1S^gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @
╕
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeShape/current_policy_network/LayerNorm_1/moments/mean*
T0*
out_type0*
_output_shapes
:
╗
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1Shape0current_policy_network/LayerNorm_1/batchnorm/mul*
out_type0*
_output_shapes
:*
T0
╟
Ygradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ShapeKgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mulMul\gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_10current_policy_network/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:         @
▓
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/SumSumGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mulYgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
к
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeReshapeGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/SumIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
С
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1Mul/current_policy_network/LayerNorm_1/moments/mean\gradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         @
╕
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1SumIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/mul_1[gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
░
Mgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1ReshapeIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Sum_1Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
·
Tgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_depsNoOpL^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeN^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1
Ю
\gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyIdentityKgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/ReshapeU^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:         *
T0
д
^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityMgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1U^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*`
_classV
TRloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/Reshape_1
ї
gradients_3/AddNAddN^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependency_1^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape_1*
N*'
_output_shapes
:         @
╣
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ShapeShape2current_policy_network/LayerNorm_1/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
У
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
┴
Wgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ShapeIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┐
Egradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mulMulgradients_3/AddN-current_policy_network/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:         @
м
Egradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/SumSumEgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mulWgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
д
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ReshapeReshapeEgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/SumGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╞
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mul_1Mul2current_policy_network/LayerNorm_1/batchnorm/Rsqrtgradients_3/AddN*
T0*'
_output_shapes
:         @
▓
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Sum_1SumGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/mul_1Ygradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Э
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1ReshapeGgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Sum_1Igradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
Ї
Rgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_depsNoOpJ^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ReshapeL^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1
Ц
Zgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependencyIdentityIgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/ReshapeS^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape*'
_output_shapes
:         
П
\gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1IdentityKgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1S^gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/Reshape_1*
_output_shapes
:@
Ь
Mgradients_3/current_policy_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad2current_policy_network/LayerNorm_1/batchnorm/RsqrtZgradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
║
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/ShapeShape3current_policy_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
М
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
┴
Wgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/ShapeIgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
Egradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/SumSumMgradients_3/current_policy_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradWgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
Igradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/ReshapeReshapeEgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/SumGgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╕
Ggradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Sum_1SumMgradients_3/current_policy_network/LayerNorm_1/batchnorm/Rsqrt_grad/RsqrtGradYgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Щ
Kgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1ReshapeGgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Sum_1Igradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
Ї
Rgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/group_depsNoOpJ^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/ReshapeL^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1
Ц
Zgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyIdentityIgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/ReshapeS^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape
Л
\gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependency_1IdentityKgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1S^gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
╞
Jgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ShapeShape<current_policy_network/LayerNorm_1/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
ъ
Igradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/SizeConst*
value	B :*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
х
Hgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/addAddEcurrent_policy_network/LayerNorm_1/moments/variance/reduction_indicesIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Size*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
э
Hgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/modFloorModHgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/addIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Size*
_output_shapes
:*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
ї
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_1Const*
_output_shapes
:*
valueB:*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0
ё
Pgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
ё
Pgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/range/deltaConst*
value	B :*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
╔
Jgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/rangeRangePgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/range/startIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/SizePgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
Ё
Ogradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
Ї
Igradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/FillFillLgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_1Ogradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Fill/value*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
ж
Rgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/DynamicStitchDynamicStitchJgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/rangeHgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/modJgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ShapeIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Fill*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
N*#
_output_shapes
:         *
T0
я
Ngradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape
И
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/MaximumMaximumRgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/DynamicStitchNgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum/y*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*#
_output_shapes
:         
ў
Mgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/floordivFloorDivJgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ShapeLgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape*
_output_shapes
:
╕
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ReshapeReshapeZgradients_3/current_policy_network/LayerNorm_1/batchnorm/add_grad/tuple/control_dependencyRgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
╗
Igradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/TileTileLgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ReshapeMgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
╚
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2Shape<current_policy_network/LayerNorm_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
┐
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_3Shape3current_policy_network/LayerNorm_1/moments/variance*
T0*
out_type0*
_output_shapes
:
ї
Jgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2
К
Igradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ProdProdLgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2Jgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Const*_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
ў
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Const_1Const*
valueB: *_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
О
Kgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Prod_1ProdLgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_3Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Const_1*
T0*_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
є
Pgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1/yConst*
value	B :*_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
·
Ngradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1MaximumKgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Prod_1Pgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2
°
Ogradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/floordiv_1FloorDivIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/ProdNgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Maximum_1*
T0*_
_classU
SQloc:@gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Shape_2*
_output_shapes
: 
╥
Igradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/CastCastOgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Я
Lgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/truedivRealDivIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/TileIgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
п
Sgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeShapecurrent_policy_network/add_1*
_output_shapes
:*
T0*
out_type0
╠
Ugradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1Shape7current_policy_network/LayerNorm_1/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
х
cgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ShapeUgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ш
Tgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/scalarConstM^gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
о
Qgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mulMulTgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/scalarLgradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
░
Qgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/subSubcurrent_policy_network/add_17current_policy_network/LayerNorm_1/moments/StopGradientM^gradients_3/current_policy_network/LayerNorm_1/moments/variance_grad/truediv*'
_output_shapes
:         @*
T0
▓
Sgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1MulQgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mulQgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/sub*'
_output_shapes
:         @*
T0
╥
Qgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/SumSumSgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1cgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╚
Ugradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeReshapeQgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/SumSgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
╓
Sgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1SumSgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/mul_1egradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╬
Wgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1ReshapeSgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Sum_1Ugradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
у
Qgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/NegNegWgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_1*'
_output_shapes
:         *
T0
Т
^gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_depsNoOpV^gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/ReshapeR^gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Neg
╞
fgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyIdentityUgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape_^gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:         @
└
hgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityQgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Neg_^gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/Neg*'
_output_shapes
:         
в
Fgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ShapeShapecurrent_policy_network/add_1*
T0*
out_type0*
_output_shapes
:
т
Egradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/SizeConst*
value	B :*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╒
Dgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/addAddAcurrent_policy_network/LayerNorm_1/moments/mean/reduction_indicesEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Size*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
▌
Dgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/modFloorModDgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/addEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Size*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
э
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_1Const*
valueB:*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
щ
Lgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/range/startConst*
_output_shapes
: *
value	B : *Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0
щ
Lgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/range/deltaConst*
value	B :*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
╡
Fgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/rangeRangeLgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/range/startEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/SizeLgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/range/delta*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
ш
Kgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape
ф
Egradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/FillFillHgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_1Kgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Fill/value*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:
О
Ngradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/DynamicStitchDynamicStitchFgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/rangeDgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/modFgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ShapeEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Fill*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
N*#
_output_shapes
:         
ч
Jgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
dtype0
°
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/MaximumMaximumNgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/DynamicStitchJgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum/y*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*#
_output_shapes
:         
ч
Igradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/floordivFloorDivFgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ShapeHgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape*
_output_shapes
:*
T0
▓
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ReshapeReshape\gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_2_grad/tuple/control_dependencyNgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
Egradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/TileTileHgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ReshapeIgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
д
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2Shapecurrent_policy_network/add_1*
T0*
out_type0*
_output_shapes
:
╖
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_3Shape/current_policy_network/LayerNorm_1/moments/mean*
T0*
out_type0*
_output_shapes
:
э
Fgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ConstConst*
_output_shapes
:*
valueB: *[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0
·
Egradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ProdProdHgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2Fgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Const*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
я
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Const_1Const*
valueB: *[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
■
Ggradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Prod_1ProdHgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_3Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Const_1*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
ы
Lgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1/yConst*
value	B :*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
ъ
Jgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1MaximumGgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Prod_1Lgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1/y*
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: 
ш
Kgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/floordiv_1FloorDivEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/ProdJgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Maximum_1*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Shape_2*
_output_shapes
: *
T0
╩
Egradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/CastCastKgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
У
Hgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/truedivRealDivEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/TileEgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/Cast*
T0*'
_output_shapes
:         @
┼
gradients_3/AddN_1AddN\gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/tuple/control_dependencyfgradients_3/current_policy_network/LayerNorm_1/moments/SquaredDifference_grad/tuple/control_dependencyHgradients_3/current_policy_network/LayerNorm_1/moments/mean_grad/truediv*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:         @
Т
3gradients_3/current_policy_network/add_1_grad/ShapeShapecurrent_policy_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

5gradients_3/current_policy_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
Е
Cgradients_3/current_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_3/current_policy_network/add_1_grad/Shape5gradients_3/current_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╤
1gradients_3/current_policy_network/add_1_grad/SumSumgradients_3/AddN_1Cgradients_3/current_policy_network/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ш
5gradients_3/current_policy_network/add_1_grad/ReshapeReshape1gradients_3/current_policy_network/add_1_grad/Sum3gradients_3/current_policy_network/add_1_grad/Shape*'
_output_shapes
:         @*
T0*
Tshape0
╒
3gradients_3/current_policy_network/add_1_grad/Sum_1Sumgradients_3/AddN_1Egradients_3/current_policy_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
с
7gradients_3/current_policy_network/add_1_grad/Reshape_1Reshape3gradients_3/current_policy_network/add_1_grad/Sum_15gradients_3/current_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
╕
>gradients_3/current_policy_network/add_1_grad/tuple/group_depsNoOp6^gradients_3/current_policy_network/add_1_grad/Reshape8^gradients_3/current_policy_network/add_1_grad/Reshape_1
╞
Fgradients_3/current_policy_network/add_1_grad/tuple/control_dependencyIdentity5gradients_3/current_policy_network/add_1_grad/Reshape?^gradients_3/current_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*H
_class>
<:loc:@gradients_3/current_policy_network/add_1_grad/Reshape
┐
Hgradients_3/current_policy_network/add_1_grad/tuple/control_dependency_1Identity7gradients_3/current_policy_network/add_1_grad/Reshape_1?^gradients_3/current_policy_network/add_1_grad/tuple/group_deps*J
_class@
><loc:@gradients_3/current_policy_network/add_1_grad/Reshape_1*
_output_shapes
:@*
T0
Ы
7gradients_3/current_policy_network/MatMul_1_grad/MatMulMatMulFgradients_3/current_policy_network/add_1_grad/tuple/control_dependency8current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
ў
9gradients_3/current_policy_network/MatMul_1_grad/MatMul_1MatMulcurrent_policy_network/TanhFgradients_3/current_policy_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
┐
Agradients_3/current_policy_network/MatMul_1_grad/tuple/group_depsNoOp8^gradients_3/current_policy_network/MatMul_1_grad/MatMul:^gradients_3/current_policy_network/MatMul_1_grad/MatMul_1
╨
Igradients_3/current_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity7gradients_3/current_policy_network/MatMul_1_grad/MatMulB^gradients_3/current_policy_network/MatMul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_3/current_policy_network/MatMul_1_grad/MatMul*'
_output_shapes
:         @
═
Kgradients_3/current_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity9gradients_3/current_policy_network/MatMul_1_grad/MatMul_1B^gradients_3/current_policy_network/MatMul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_3/current_policy_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
█
5gradients_3/current_policy_network/Tanh_grad/TanhGradTanhGradcurrent_policy_network/TanhIgradients_3/current_policy_network/MatMul_1_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
╖
Ggradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeShape0current_policy_network/LayerNorm/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
╖
Igradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1Shape.current_policy_network/LayerNorm/batchnorm/sub*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/ShapeIgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ь
Egradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/SumSum5gradients_3/current_policy_network/Tanh_grad/TanhGradWgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
д
Igradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeReshapeEgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/SumGgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
а
Ggradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1Sum5gradients_3/current_policy_network/Tanh_grad/TanhGradYgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
к
Kgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1ReshapeGgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Sum_1Igradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_depsNoOpJ^gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeL^gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1
Ц
Zgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependencyIdentityIgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/ReshapeS^gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape*'
_output_shapes
:         @*
T0
Ь
\gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1IdentityKgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1S^gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:         @
б
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeShapecurrent_policy_network/add*
out_type0*
_output_shapes
:*
T0
╖
Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1Shape.current_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ShapeIgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
К
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mulMulZgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency.current_policy_network/LayerNorm/batchnorm/mul*'
_output_shapes
:         @*
T0
м
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/SumSumEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mulWgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeReshapeEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/SumGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
°
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1Mulcurrent_policy_network/addZgradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
▓
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1SumGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/mul_1Ygradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
к
Kgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1ReshapeGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Sum_1Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_depsNoOpJ^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeL^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
Ц
Zgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencyIdentityIgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/ReshapeS^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:         @
Ь
\gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityKgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1S^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
П
Egradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@
╖
Ggradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1Shape0current_policy_network/LayerNorm/batchnorm/mul_2*
T0*
out_type0*
_output_shapes
:
╗
Ugradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/ShapeGgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┐
Cgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/SumSum\gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Ugradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
Ggradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeReshapeCgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/SumEgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:@
├
Egradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1Sum\gradients_3/current_policy_network/LayerNorm/batchnorm/add_1_grad/tuple/control_dependency_1Wgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┤
Cgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/NegNegEgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
в
Igradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1ReshapeCgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/NegGgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Shape_1*'
_output_shapes
:         @*
T0*
Tshape0
ю
Pgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_depsNoOpH^gradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeJ^gradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1
Б
Xgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependencyIdentityGgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/ReshapeQ^gradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape*
_output_shapes
:@
Ф
Zgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1IdentityIgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1Q^gradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:         @
┤
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeShape-current_policy_network/LayerNorm/moments/mean*
out_type0*
_output_shapes
:*
T0
╖
Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1Shape.current_policy_network/LayerNorm/batchnorm/mul*
T0*
out_type0*
_output_shapes
:
┴
Wgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ShapeIgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
К
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mulMulZgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1.current_policy_network/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:         @
м
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/SumSumEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mulWgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeReshapeEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/SumGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
Л
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Mul-current_policy_network/LayerNorm/moments/meanZgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         @
▓
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1SumGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/mul_1Ygradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
к
Kgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1ReshapeGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Sum_1Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         @
Ї
Rgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_depsNoOpJ^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeL^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1
Ц
Zgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyIdentityIgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/ReshapeS^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape*'
_output_shapes
:         *
T0
Ь
\gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityKgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1S^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/group_deps*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/Reshape_1*'
_output_shapes
:         @*
T0
ё
gradients_3/AddN_2AddN\gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependency_1\gradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*'
_output_shapes
:         @*
T0*^
_classT
RPloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape_1
╡
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/ShapeShape0current_policy_network/LayerNorm/batchnorm/Rsqrt*
T0*
out_type0*
_output_shapes
:
С
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
╗
Ugradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/ShapeGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╜
Cgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/mulMulgradients_3/AddN_2+current_policy_network/LayerNorm/gamma/read*'
_output_shapes
:         @*
T0
ж
Cgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/SumSumCgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/mulUgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeReshapeCgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/SumEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
─
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Mul0current_policy_network/LayerNorm/batchnorm/Rsqrtgradients_3/AddN_2*'
_output_shapes
:         @*
T0
м
Egradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1SumEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/mul_1Wgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
Igradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1ReshapeEgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Sum_1Ggradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
ю
Pgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_depsNoOpH^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeJ^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
О
Xgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependencyIdentityGgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/ReshapeQ^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape*'
_output_shapes
:         
З
Zgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1IdentityIgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1Q^gradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:@*
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/Reshape_1
Ц
Kgradients_3/current_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad0current_policy_network/LayerNorm/batchnorm/RsqrtXgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
╢
Egradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/ShapeShape1current_policy_network/LayerNorm/moments/variance*
T0*
out_type0*
_output_shapes
:
К
Ggradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╗
Ugradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/ShapeGgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
о
Cgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/SumSumKgradients_3/current_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradUgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ю
Ggradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/ReshapeReshapeCgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/SumEgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
▓
Egradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Sum_1SumKgradients_3/current_policy_network/LayerNorm/batchnorm/Rsqrt_grad/RsqrtGradWgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
Igradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1ReshapeEgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Sum_1Ggradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ю
Pgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_depsNoOpH^gradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/ReshapeJ^gradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1
О
Xgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyIdentityGgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/ReshapeQ^gradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*Z
_classP
NLloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape
Г
Zgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependency_1IdentityIgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1Q^gradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/group_deps*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/Reshape_1*
_output_shapes
: *
T0
┬
Hgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ShapeShape:current_policy_network/LayerNorm/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0
ц
Ggradients_3/current_policy_network/LayerNorm/moments/variance_grad/SizeConst*
value	B :*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
▌
Fgradients_3/current_policy_network/LayerNorm/moments/variance_grad/addAddCcurrent_policy_network/LayerNorm/moments/variance/reduction_indicesGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Size*
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
х
Fgradients_3/current_policy_network/LayerNorm/moments/variance_grad/modFloorModFgradients_3/current_policy_network/LayerNorm/moments/variance_grad/addGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Size*
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
ё
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape
э
Ngradients_3/current_policy_network/LayerNorm/moments/variance_grad/range/startConst*
value	B : *[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
э
Ngradients_3/current_policy_network/LayerNorm/moments/variance_grad/range/deltaConst*
value	B :*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
┐
Hgradients_3/current_policy_network/LayerNorm/moments/variance_grad/rangeRangeNgradients_3/current_policy_network/LayerNorm/moments/variance_grad/range/startGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/SizeNgradients_3/current_policy_network/LayerNorm/moments/variance_grad/range/delta*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
ь
Mgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape
ь
Ggradients_3/current_policy_network/LayerNorm/moments/variance_grad/FillFillJgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_1Mgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Fill/value*
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
Ъ
Pgradients_3/current_policy_network/LayerNorm/moments/variance_grad/DynamicStitchDynamicStitchHgradients_3/current_policy_network/LayerNorm/moments/variance_grad/rangeFgradients_3/current_policy_network/LayerNorm/moments/variance_grad/modHgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ShapeGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Fill*
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
N*#
_output_shapes
:         
ы
Lgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum/yConst*
value	B :*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
А
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/MaximumMaximumPgradients_3/current_policy_network/LayerNorm/moments/variance_grad/DynamicStitchLgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum/y*#
_output_shapes
:         *
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape
я
Kgradients_3/current_policy_network/LayerNorm/moments/variance_grad/floordivFloorDivHgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ShapeJgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum*
T0*[
_classQ
OMloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape*
_output_shapes
:
▓
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ReshapeReshapeXgradients_3/current_policy_network/LayerNorm/batchnorm/add_grad/tuple/control_dependencyPgradients_3/current_policy_network/LayerNorm/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
╡
Ggradients_3/current_policy_network/LayerNorm/moments/variance_grad/TileTileJgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ReshapeKgradients_3/current_policy_network/LayerNorm/moments/variance_grad/floordiv*0
_output_shapes
:                  *

Tmultiples0*
T0
─
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2Shape:current_policy_network/LayerNorm/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0
╗
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_3Shape1current_policy_network/LayerNorm/moments/variance*
_output_shapes
:*
T0*
out_type0
ё
Hgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ConstConst*
valueB: *]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
В
Ggradients_3/current_policy_network/LayerNorm/moments/variance_grad/ProdProdJgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2Hgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Const*
	keep_dims( *

Tidx0*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
є
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Const_1Const*
_output_shapes
:*
valueB: *]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0
Ж
Igradients_3/current_policy_network/LayerNorm/moments/variance_grad/Prod_1ProdJgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_3Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Const_1*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
я
Ngradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1/yConst*
value	B :*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
Є
Lgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1MaximumIgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Prod_1Ngradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1/y*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
Ё
Mgradients_3/current_policy_network/LayerNorm/moments/variance_grad/floordiv_1FloorDivGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/ProdLgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Maximum_1*
T0*]
_classS
QOloc:@gradients_3/current_policy_network/LayerNorm/moments/variance_grad/Shape_2*
_output_shapes
: 
╬
Ggradients_3/current_policy_network/LayerNorm/moments/variance_grad/CastCastMgradients_3/current_policy_network/LayerNorm/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Щ
Jgradients_3/current_policy_network/LayerNorm/moments/variance_grad/truedivRealDivGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/TileGgradients_3/current_policy_network/LayerNorm/moments/variance_grad/Cast*
T0*'
_output_shapes
:         @
л
Qgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeShapecurrent_policy_network/add*
T0*
out_type0*
_output_shapes
:
╚
Sgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1Shape5current_policy_network/LayerNorm/moments/StopGradient*
T0*
out_type0*
_output_shapes
:
▀
agradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ShapeSgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ф
Rgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarConstK^gradients_3/current_policy_network/LayerNorm/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
и
Ogradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mulMulRgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/scalarJgradients_3/current_policy_network/LayerNorm/moments/variance_grad/truediv*
T0*'
_output_shapes
:         @
и
Ogradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/subSubcurrent_policy_network/add5current_policy_network/LayerNorm/moments/StopGradientK^gradients_3/current_policy_network/LayerNorm/moments/variance_grad/truediv*'
_output_shapes
:         @*
T0
м
Qgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1MulOgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mulOgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         @
╠
Ogradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/SumSumQgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1agradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┬
Sgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeReshapeOgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/SumQgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
╨
Qgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1SumQgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/mul_1cgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╚
Ugradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1ReshapeQgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Sum_1Sgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
▀
Ogradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/NegNegUgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
М
\gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_depsNoOpT^gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/ReshapeP^gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg
╛
dgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyIdentitySgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape]^gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*f
_class\
ZXloc:@gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Reshape
╕
fgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityOgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg]^gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/group_deps*b
_classX
VTloc:@gradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/Neg*'
_output_shapes
:         *
T0
Ю
Dgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ShapeShapecurrent_policy_network/add*
_output_shapes
:*
T0*
out_type0
▐
Cgradients_3/current_policy_network/LayerNorm/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape
═
Bgradients_3/current_policy_network/LayerNorm/moments/mean_grad/addAdd?current_policy_network/LayerNorm/moments/mean/reduction_indicesCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Size*
T0*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
╒
Bgradients_3/current_policy_network/LayerNorm/moments/mean_grad/modFloorModBgradients_3/current_policy_network/LayerNorm/moments/mean_grad/addCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Size*
T0*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:
щ
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_1Const*
_output_shapes
:*
valueB:*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0
х
Jgradients_3/current_policy_network/LayerNorm/moments/mean_grad/range/startConst*
value	B : *W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
х
Jgradients_3/current_policy_network/LayerNorm/moments/mean_grad/range/deltaConst*
value	B :*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
л
Dgradients_3/current_policy_network/LayerNorm/moments/mean_grad/rangeRangeJgradients_3/current_policy_network/LayerNorm/moments/mean_grad/range/startCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/SizeJgradients_3/current_policy_network/LayerNorm/moments/mean_grad/range/delta*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
ф
Igradients_3/current_policy_network/LayerNorm/moments/mean_grad/Fill/valueConst*
value	B :*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
▄
Cgradients_3/current_policy_network/LayerNorm/moments/mean_grad/FillFillFgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_1Igradients_3/current_policy_network/LayerNorm/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape
В
Lgradients_3/current_policy_network/LayerNorm/moments/mean_grad/DynamicStitchDynamicStitchDgradients_3/current_policy_network/LayerNorm/moments/mean_grad/rangeBgradients_3/current_policy_network/LayerNorm/moments/mean_grad/modDgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ShapeCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Fill*
T0*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
N*#
_output_shapes
:         
у
Hgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum/yConst*
value	B :*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
Ё
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/MaximumMaximumLgradients_3/current_policy_network/LayerNorm/moments/mean_grad/DynamicStitchHgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum/y*
T0*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape*#
_output_shapes
:         
▀
Ggradients_3/current_policy_network/LayerNorm/moments/mean_grad/floordivFloorDivDgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ShapeFgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum*
_output_shapes
:*
T0*W
_classM
KIloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape
м
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ReshapeReshapeZgradients_3/current_policy_network/LayerNorm/batchnorm/mul_2_grad/tuple/control_dependencyLgradients_3/current_policy_network/LayerNorm/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
й
Cgradients_3/current_policy_network/LayerNorm/moments/mean_grad/TileTileFgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ReshapeGgradients_3/current_policy_network/LayerNorm/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
а
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2Shapecurrent_policy_network/add*
_output_shapes
:*
T0*
out_type0
│
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_3Shape-current_policy_network/LayerNorm/moments/mean*
_output_shapes
:*
T0*
out_type0
щ
Dgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ConstConst*
valueB: *Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
Є
Cgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ProdProdFgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2Dgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Const*
	keep_dims( *

Tidx0*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
ы
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Const_1Const*
valueB: *Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
Ў
Egradients_3/current_policy_network/LayerNorm/moments/mean_grad/Prod_1ProdFgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_3Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Const_1*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
ч
Jgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2
т
Hgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1MaximumEgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Prod_1Jgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1/y*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
р
Igradients_3/current_policy_network/LayerNorm/moments/mean_grad/floordiv_1FloorDivCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/ProdHgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Maximum_1*
T0*Y
_classO
MKloc:@gradients_3/current_policy_network/LayerNorm/moments/mean_grad/Shape_2*
_output_shapes
: 
╞
Cgradients_3/current_policy_network/LayerNorm/moments/mean_grad/CastCastIgradients_3/current_policy_network/LayerNorm/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Н
Fgradients_3/current_policy_network/LayerNorm/moments/mean_grad/truedivRealDivCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/TileCgradients_3/current_policy_network/LayerNorm/moments/mean_grad/Cast*'
_output_shapes
:         @*
T0
╜
gradients_3/AddN_3AddNZgradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/tuple/control_dependencydgradients_3/current_policy_network/LayerNorm/moments/SquaredDifference_grad/tuple/control_dependencyFgradients_3/current_policy_network/LayerNorm/moments/mean_grad/truediv*
T0*\
_classR
PNloc:@gradients_3/current_policy_network/LayerNorm/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:         @
О
1gradients_3/current_policy_network/add_grad/ShapeShapecurrent_policy_network/MatMul*
T0*
out_type0*
_output_shapes
:
}
3gradients_3/current_policy_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
 
Agradients_3/current_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_3/current_policy_network/add_grad/Shape3gradients_3/current_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
═
/gradients_3/current_policy_network/add_grad/SumSumgradients_3/AddN_3Agradients_3/current_policy_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
т
3gradients_3/current_policy_network/add_grad/ReshapeReshape/gradients_3/current_policy_network/add_grad/Sum1gradients_3/current_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         @
╤
1gradients_3/current_policy_network/add_grad/Sum_1Sumgradients_3/AddN_3Cgradients_3/current_policy_network/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
█
5gradients_3/current_policy_network/add_grad/Reshape_1Reshape1gradients_3/current_policy_network/add_grad/Sum_13gradients_3/current_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
▓
<gradients_3/current_policy_network/add_grad/tuple/group_depsNoOp4^gradients_3/current_policy_network/add_grad/Reshape6^gradients_3/current_policy_network/add_grad/Reshape_1
╛
Dgradients_3/current_policy_network/add_grad/tuple/control_dependencyIdentity3gradients_3/current_policy_network/add_grad/Reshape=^gradients_3/current_policy_network/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_3/current_policy_network/add_grad/Reshape*'
_output_shapes
:         @
╖
Fgradients_3/current_policy_network/add_grad/tuple/control_dependency_1Identity5gradients_3/current_policy_network/add_grad/Reshape_1=^gradients_3/current_policy_network/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/current_policy_network/add_grad/Reshape_1*
_output_shapes
:@
Ч
5gradients_3/current_policy_network/MatMul_grad/MatMulMatMulDgradients_3/current_policy_network/add_grad/tuple/control_dependency8current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
ц
7gradients_3/current_policy_network/MatMul_grad/MatMul_1MatMulobservations_2Dgradients_3/current_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
╣
?gradients_3/current_policy_network/MatMul_grad/tuple/group_depsNoOp6^gradients_3/current_policy_network/MatMul_grad/MatMul8^gradients_3/current_policy_network/MatMul_grad/MatMul_1
╚
Ggradients_3/current_policy_network/MatMul_grad/tuple/control_dependencyIdentity5gradients_3/current_policy_network/MatMul_grad/MatMul@^gradients_3/current_policy_network/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/current_policy_network/MatMul_grad/MatMul*'
_output_shapes
:         
┼
Igradients_3/current_policy_network/MatMul_grad/tuple/control_dependency_1Identity7gradients_3/current_policy_network/MatMul_grad/MatMul_1@^gradients_3/current_policy_network/MatMul_grad/tuple/group_deps*J
_class@
><loc:@gradients_3/current_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:@*
T0
Ъ
beta1_power_3/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
dtype0
л
beta1_power_3
VariableV2*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
╬
beta1_power_3/AssignAssignbeta1_power_3beta1_power_3/initial_value*
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
И
beta1_power_3/readIdentitybeta1_power_3*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
: *
T0
Ъ
beta2_power_3/initial_valueConst*
_output_shapes
: *
valueB
 *w╛?*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
dtype0
л
beta2_power_3
VariableV2*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
╬
beta2_power_3/AssignAssignbeta2_power_3beta2_power_3/initial_value*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
И
beta2_power_3/readIdentitybeta2_power_3*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
: 
щ
Lcurrent_policy_network/current_policy_network/fc0/w/Adam_2/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ў
:current_policy_network/current_policy_network/fc0/w/Adam_2
VariableV2*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@*
dtype0
я
Acurrent_policy_network/current_policy_network/fc0/w/Adam_2/AssignAssign:current_policy_network/current_policy_network/fc0/w/Adam_2Lcurrent_policy_network/current_policy_network/fc0/w/Adam_2/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
validate_shape(
°
?current_policy_network/current_policy_network/fc0/w/Adam_2/readIdentity:current_policy_network/current_policy_network/fc0/w/Adam_2*
_output_shapes

:@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w
щ
Lcurrent_policy_network/current_policy_network/fc0/w/Adam_3/Initializer/zerosConst*
_output_shapes

:@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0
Ў
:current_policy_network/current_policy_network/fc0/w/Adam_3
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w
я
Acurrent_policy_network/current_policy_network/fc0/w/Adam_3/AssignAssign:current_policy_network/current_policy_network/fc0/w/Adam_3Lcurrent_policy_network/current_policy_network/fc0/w/Adam_3/Initializer/zeros*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
°
?current_policy_network/current_policy_network/fc0/w/Adam_3/readIdentity:current_policy_network/current_policy_network/fc0/w/Adam_3*
_output_shapes

:@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w
с
Lcurrent_policy_network/current_policy_network/fc0/b/Adam_2/Initializer/zerosConst*
_output_shapes
:@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0
ю
:current_policy_network/current_policy_network/fc0/b/Adam_2
VariableV2*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
	container *
shape:@*
dtype0
ы
Acurrent_policy_network/current_policy_network/fc0/b/Adam_2/AssignAssign:current_policy_network/current_policy_network/fc0/b/Adam_2Lcurrent_policy_network/current_policy_network/fc0/b/Adam_2/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
Ї
?current_policy_network/current_policy_network/fc0/b/Adam_2/readIdentity:current_policy_network/current_policy_network/fc0/b/Adam_2*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
с
Lcurrent_policy_network/current_policy_network/fc0/b/Adam_3/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ю
:current_policy_network/current_policy_network/fc0/b/Adam_3
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
	container 
ы
Acurrent_policy_network/current_policy_network/fc0/b/Adam_3/AssignAssign:current_policy_network/current_policy_network/fc0/b/Adam_3Lcurrent_policy_network/current_policy_network/fc0/b/Adam_3/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
Ї
?current_policy_network/current_policy_network/fc0/b/Adam_3/readIdentity:current_policy_network/current_policy_network/fc0/b/Adam_3*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@*
T0
┼
>current_policy_network/LayerNorm/beta/Adam_2/Initializer/zerosConst*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╥
,current_policy_network/LayerNorm/beta/Adam_2
VariableV2*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape:@*
dtype0
│
3current_policy_network/LayerNorm/beta/Adam_2/AssignAssign,current_policy_network/LayerNorm/beta/Adam_2>current_policy_network/LayerNorm/beta/Adam_2/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
╩
1current_policy_network/LayerNorm/beta/Adam_2/readIdentity,current_policy_network/LayerNorm/beta/Adam_2*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
:@
┼
>current_policy_network/LayerNorm/beta/Adam_3/Initializer/zerosConst*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╥
,current_policy_network/LayerNorm/beta/Adam_3
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
	container *
shape:@
│
3current_policy_network/LayerNorm/beta/Adam_3/AssignAssign,current_policy_network/LayerNorm/beta/Adam_3>current_policy_network/LayerNorm/beta/Adam_3/Initializer/zeros*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
╩
1current_policy_network/LayerNorm/beta/Adam_3/readIdentity,current_policy_network/LayerNorm/beta/Adam_3*
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
:@
╟
?current_policy_network/LayerNorm/gamma/Adam_2/Initializer/zerosConst*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╘
-current_policy_network/LayerNorm/gamma/Adam_2
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_policy_network/LayerNorm/gamma
╖
4current_policy_network/LayerNorm/gamma/Adam_2/AssignAssign-current_policy_network/LayerNorm/gamma/Adam_2?current_policy_network/LayerNorm/gamma/Adam_2/Initializer/zeros*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
═
2current_policy_network/LayerNorm/gamma/Adam_2/readIdentity-current_policy_network/LayerNorm/gamma/Adam_2*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
_output_shapes
:@
╟
?current_policy_network/LayerNorm/gamma/Adam_3/Initializer/zerosConst*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╘
-current_policy_network/LayerNorm/gamma/Adam_3
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
	container *
shape:@
╖
4current_policy_network/LayerNorm/gamma/Adam_3/AssignAssign-current_policy_network/LayerNorm/gamma/Adam_3?current_policy_network/LayerNorm/gamma/Adam_3/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
═
2current_policy_network/LayerNorm/gamma/Adam_3/readIdentity-current_policy_network/LayerNorm/gamma/Adam_3*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
_output_shapes
:@
щ
Lcurrent_policy_network/current_policy_network/fc1/w/Adam_2/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
Ў
:current_policy_network/current_policy_network/fc1/w/Adam_2
VariableV2*
_output_shapes

:@@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@*
dtype0
я
Acurrent_policy_network/current_policy_network/fc1/w/Adam_2/AssignAssign:current_policy_network/current_policy_network/fc1/w/Adam_2Lcurrent_policy_network/current_policy_network/fc1/w/Adam_2/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
°
?current_policy_network/current_policy_network/fc1/w/Adam_2/readIdentity:current_policy_network/current_policy_network/fc1/w/Adam_2*
_output_shapes

:@@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w
щ
Lcurrent_policy_network/current_policy_network/fc1/w/Adam_3/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
Ў
:current_policy_network/current_policy_network/fc1/w/Adam_3
VariableV2*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
я
Acurrent_policy_network/current_policy_network/fc1/w/Adam_3/AssignAssign:current_policy_network/current_policy_network/fc1/w/Adam_3Lcurrent_policy_network/current_policy_network/fc1/w/Adam_3/Initializer/zeros*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(
°
?current_policy_network/current_policy_network/fc1/w/Adam_3/readIdentity:current_policy_network/current_policy_network/fc1/w/Adam_3*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
с
Lcurrent_policy_network/current_policy_network/fc1/b/Adam_2/Initializer/zerosConst*
dtype0*
_output_shapes
:@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
valueB@*    
ю
:current_policy_network/current_policy_network/fc1/b/Adam_2
VariableV2*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
ы
Acurrent_policy_network/current_policy_network/fc1/b/Adam_2/AssignAssign:current_policy_network/current_policy_network/fc1/b/Adam_2Lcurrent_policy_network/current_policy_network/fc1/b/Adam_2/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
Ї
?current_policy_network/current_policy_network/fc1/b/Adam_2/readIdentity:current_policy_network/current_policy_network/fc1/b/Adam_2*
_output_shapes
:@*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b
с
Lcurrent_policy_network/current_policy_network/fc1/b/Adam_3/Initializer/zerosConst*
dtype0*
_output_shapes
:@*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
valueB@*    
ю
:current_policy_network/current_policy_network/fc1/b/Adam_3
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
	container 
ы
Acurrent_policy_network/current_policy_network/fc1/b/Adam_3/AssignAssign:current_policy_network/current_policy_network/fc1/b/Adam_3Lcurrent_policy_network/current_policy_network/fc1/b/Adam_3/Initializer/zeros*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ї
?current_policy_network/current_policy_network/fc1/b/Adam_3/readIdentity:current_policy_network/current_policy_network/fc1/b/Adam_3*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
╔
@current_policy_network/LayerNorm_1/beta/Adam_2/Initializer/zerosConst*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╓
.current_policy_network/LayerNorm_1/beta/Adam_2
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
	container *
shape:@
╗
5current_policy_network/LayerNorm_1/beta/Adam_2/AssignAssign.current_policy_network/LayerNorm_1/beta/Adam_2@current_policy_network/LayerNorm_1/beta/Adam_2/Initializer/zeros*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
╨
3current_policy_network/LayerNorm_1/beta/Adam_2/readIdentity.current_policy_network/LayerNorm_1/beta/Adam_2*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
_output_shapes
:@
╔
@current_policy_network/LayerNorm_1/beta/Adam_3/Initializer/zerosConst*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
╓
.current_policy_network/LayerNorm_1/beta/Adam_3
VariableV2*
shared_name *:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
╗
5current_policy_network/LayerNorm_1/beta/Adam_3/AssignAssign.current_policy_network/LayerNorm_1/beta/Adam_3@current_policy_network/LayerNorm_1/beta/Adam_3/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
validate_shape(
╨
3current_policy_network/LayerNorm_1/beta/Adam_3/readIdentity.current_policy_network/LayerNorm_1/beta/Adam_3*
_output_shapes
:@*
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta
╦
Acurrent_policy_network/LayerNorm_1/gamma/Adam_2/Initializer/zerosConst*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╪
/current_policy_network/LayerNorm_1/gamma/Adam_2
VariableV2*
shared_name *;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
┐
6current_policy_network/LayerNorm_1/gamma/Adam_2/AssignAssign/current_policy_network/LayerNorm_1/gamma/Adam_2Acurrent_policy_network/LayerNorm_1/gamma/Adam_2/Initializer/zeros*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
╙
4current_policy_network/LayerNorm_1/gamma/Adam_2/readIdentity/current_policy_network/LayerNorm_1/gamma/Adam_2*
_output_shapes
:@*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma
╦
Acurrent_policy_network/LayerNorm_1/gamma/Adam_3/Initializer/zerosConst*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
╪
/current_policy_network/LayerNorm_1/gamma/Adam_3
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
	container *
shape:@
┐
6current_policy_network/LayerNorm_1/gamma/Adam_3/AssignAssign/current_policy_network/LayerNorm_1/gamma/Adam_3Acurrent_policy_network/LayerNorm_1/gamma/Adam_3/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
╙
4current_policy_network/LayerNorm_1/gamma/Adam_3/readIdentity/current_policy_network/LayerNorm_1/gamma/Adam_3*
_output_shapes
:@*
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma
щ
Lcurrent_policy_network/current_policy_network/out/w/Adam_2/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
Ў
:current_policy_network/current_policy_network/out/w/Adam_2
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/w
я
Acurrent_policy_network/current_policy_network/out/w/Adam_2/AssignAssign:current_policy_network/current_policy_network/out/w/Adam_2Lcurrent_policy_network/current_policy_network/out/w/Adam_2/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
°
?current_policy_network/current_policy_network/out/w/Adam_2/readIdentity:current_policy_network/current_policy_network/out/w/Adam_2*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
_output_shapes

:@
щ
Lcurrent_policy_network/current_policy_network/out/w/Adam_3/Initializer/zerosConst*
dtype0*
_output_shapes

:@*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
valueB@*    
Ў
:current_policy_network/current_policy_network/out/w/Adam_3
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
	container *
shape
:@
я
Acurrent_policy_network/current_policy_network/out/w/Adam_3/AssignAssign:current_policy_network/current_policy_network/out/w/Adam_3Lcurrent_policy_network/current_policy_network/out/w/Adam_3/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
validate_shape(
°
?current_policy_network/current_policy_network/out/w/Adam_3/readIdentity:current_policy_network/current_policy_network/out/w/Adam_3*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
_output_shapes

:@
с
Lcurrent_policy_network/current_policy_network/out/b/Adam_2/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ю
:current_policy_network/current_policy_network/out/b/Adam_2
VariableV2*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ы
Acurrent_policy_network/current_policy_network/out/b/Adam_2/AssignAssign:current_policy_network/current_policy_network/out/b/Adam_2Lcurrent_policy_network/current_policy_network/out/b/Adam_2/Initializer/zeros*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ї
?current_policy_network/current_policy_network/out/b/Adam_2/readIdentity:current_policy_network/current_policy_network/out/b/Adam_2*
_output_shapes
:*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b
с
Lcurrent_policy_network/current_policy_network/out/b/Adam_3/Initializer/zerosConst*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ю
:current_policy_network/current_policy_network/out/b/Adam_3
VariableV2*
dtype0*
_output_shapes
:*
shared_name *F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
	container *
shape:
ы
Acurrent_policy_network/current_policy_network/out/b/Adam_3/AssignAssign:current_policy_network/current_policy_network/out/b/Adam_3Lcurrent_policy_network/current_policy_network/out/b/Adam_3/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
Ї
?current_policy_network/current_policy_network/out/b/Adam_3/readIdentity:current_policy_network/current_policy_network/out/b/Adam_3*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
_output_shapes
:
Q
Adam_3/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_3/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
S
Adam_3/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
═
KAdam_3/update_current_policy_network/current_policy_network/fc0/w/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc0/w:current_policy_network/current_policy_network/fc0/w/Adam_2:current_policy_network/current_policy_network/fc0/w/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonIgradients_3/current_policy_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:@
╞
KAdam_3/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc0/b:current_policy_network/current_policy_network/fc0/b/Adam_2:current_policy_network/current_policy_network/fc0/b/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonFgradients_3/current_policy_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:@
Т
=Adam_3/update_current_policy_network/LayerNorm/beta/ApplyAdam	ApplyAdam%current_policy_network/LayerNorm/beta,current_policy_network/LayerNorm/beta/Adam_2,current_policy_network/LayerNorm/beta/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonXgradients_3/current_policy_network/LayerNorm/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
use_nesterov( *
_output_shapes
:@
Щ
>Adam_3/update_current_policy_network/LayerNorm/gamma/ApplyAdam	ApplyAdam&current_policy_network/LayerNorm/gamma-current_policy_network/LayerNorm/gamma/Adam_2-current_policy_network/LayerNorm/gamma/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonZgradients_3/current_policy_network/LayerNorm/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
use_nesterov( 
╧
KAdam_3/update_current_policy_network/current_policy_network/fc1/w/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc1/w:current_policy_network/current_policy_network/fc1/w/Adam_2:current_policy_network/current_policy_network/fc1/w/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonKgradients_3/current_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
use_nesterov( *
_output_shapes

:@@
╚
KAdam_3/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/fc1/b:current_policy_network/current_policy_network/fc1/b/Adam_2:current_policy_network/current_policy_network/fc1/b/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonHgradients_3/current_policy_network/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b
Ю
?Adam_3/update_current_policy_network/LayerNorm_1/beta/ApplyAdam	ApplyAdam'current_policy_network/LayerNorm_1/beta.current_policy_network/LayerNorm_1/beta/Adam_2.current_policy_network/LayerNorm_1/beta/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonZgradients_3/current_policy_network/LayerNorm_1/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
use_nesterov( *
_output_shapes
:@
е
@Adam_3/update_current_policy_network/LayerNorm_1/gamma/ApplyAdam	ApplyAdam(current_policy_network/LayerNorm_1/gamma/current_policy_network/LayerNorm_1/gamma/Adam_2/current_policy_network/LayerNorm_1/gamma/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilon\gradients_3/current_policy_network/LayerNorm_1/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma
╧
KAdam_3/update_current_policy_network/current_policy_network/out/w/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/out/w:current_policy_network/current_policy_network/out/w/Adam_2:current_policy_network/current_policy_network/out/w/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonKgradients_3/current_policy_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
use_nesterov( *
_output_shapes

:@
╚
KAdam_3/update_current_policy_network/current_policy_network/out/b/ApplyAdam	ApplyAdam3current_policy_network/current_policy_network/out/b:current_policy_network/current_policy_network/out/b/Adam_2:current_policy_network/current_policy_network/out/b/Adam_3beta1_power_3/readbeta2_power_3/readlearning_rate_2Adam_3/beta1Adam_3/beta2Adam_3/epsilonHgradients_3/current_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
use_nesterov( *
_output_shapes
:
ш

Adam_3/mulMulbeta1_power_3/readAdam_3/beta1L^Adam_3/update_current_policy_network/current_policy_network/fc0/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam>^Adam_3/update_current_policy_network/LayerNorm/beta/ApplyAdam?^Adam_3/update_current_policy_network/LayerNorm/gamma/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc1/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam@^Adam_3/update_current_policy_network/LayerNorm_1/beta/ApplyAdamA^Adam_3/update_current_policy_network/LayerNorm_1/gamma/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/out/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/out/b/ApplyAdam*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
_output_shapes
: *
T0
╢
Adam_3/AssignAssignbeta1_power_3
Adam_3/mul*
use_locking( *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
ъ
Adam_3/mul_1Mulbeta2_power_3/readAdam_3/beta2L^Adam_3/update_current_policy_network/current_policy_network/fc0/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam>^Adam_3/update_current_policy_network/LayerNorm/beta/ApplyAdam?^Adam_3/update_current_policy_network/LayerNorm/gamma/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc1/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam@^Adam_3/update_current_policy_network/LayerNorm_1/beta/ApplyAdamA^Adam_3/update_current_policy_network/LayerNorm_1/gamma/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/out/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta
║
Adam_3/Assign_1Assignbeta2_power_3Adam_3/mul_1*
use_locking( *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
: 
К
Adam_3NoOpL^Adam_3/update_current_policy_network/current_policy_network/fc0/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc0/b/ApplyAdam>^Adam_3/update_current_policy_network/LayerNorm/beta/ApplyAdam?^Adam_3/update_current_policy_network/LayerNorm/gamma/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc1/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/fc1/b/ApplyAdam@^Adam_3/update_current_policy_network/LayerNorm_1/beta/ApplyAdamA^Adam_3/update_current_policy_network/LayerNorm_1/gamma/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/out/w/ApplyAdamL^Adam_3/update_current_policy_network/current_policy_network/out/b/ApplyAdam^Adam_3/Assign^Adam_3/Assign_1
ь
	Assign_80Assign$target_policy_network/LayerNorm/beta*current_policy_network/LayerNorm/beta/read*
use_locking( *
T0*7
_class-
+)loc:@target_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
я
	Assign_81Assign%target_policy_network/LayerNorm/gamma+current_policy_network/LayerNorm/gamma/read*
use_locking( *
T0*8
_class.
,*loc:@target_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
Є
	Assign_82Assign&target_policy_network/LayerNorm_1/beta,current_policy_network/LayerNorm_1/beta/read*9
_class/
-+loc:@target_policy_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
ї
	Assign_83Assign'target_policy_network/LayerNorm_1/gamma-current_policy_network/LayerNorm_1/gamma/read*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@target_policy_network/LayerNorm_1/gamma*
validate_shape(
Ф
	Assign_84Assign1target_policy_network/target_policy_network/fc0/b8current_policy_network/current_policy_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/b*
validate_shape(
Ш
	Assign_85Assign1target_policy_network/target_policy_network/fc0/w8current_policy_network/current_policy_network/fc0/w/read*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
Ф
	Assign_86Assign1target_policy_network/target_policy_network/fc1/b8current_policy_network/current_policy_network/fc1/b/read*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/b*
validate_shape(
Ш
	Assign_87Assign1target_policy_network/target_policy_network/fc1/w8current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
Ф
	Assign_88Assign1target_policy_network/target_policy_network/out/b8current_policy_network/current_policy_network/out/b/read*D
_class:
86loc:@target_policy_network/target_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
Ш
	Assign_89Assign1target_policy_network/target_policy_network/out/w8current_policy_network/current_policy_network/out/w/read*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( 
М
group_deps_6NoOp
^Assign_80
^Assign_81
^Assign_82
^Assign_83
^Assign_84
^Assign_85
^Assign_86
^Assign_87
^Assign_88
^Assign_89
ш
	Assign_90Assign"last_policy_network/LayerNorm/beta*current_policy_network/LayerNorm/beta/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*5
_class+
)'loc:@last_policy_network/LayerNorm/beta
ы
	Assign_91Assign#last_policy_network/LayerNorm/gamma+current_policy_network/LayerNorm/gamma/read*
use_locking( *
T0*6
_class,
*(loc:@last_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
ю
	Assign_92Assign$last_policy_network/LayerNorm_1/beta,current_policy_network/LayerNorm_1/beta/read*
use_locking( *
T0*7
_class-
+)loc:@last_policy_network/LayerNorm_1/beta*
validate_shape(*
_output_shapes
:@
ё
	Assign_93Assign%last_policy_network/LayerNorm_1/gamma-current_policy_network/LayerNorm_1/gamma/read*
_output_shapes
:@*
use_locking( *
T0*8
_class.
,*loc:@last_policy_network/LayerNorm_1/gamma*
validate_shape(
М
	Assign_94Assign-last_policy_network/last_policy_network/fc0/b8current_policy_network/current_policy_network/fc0/b/read*
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( 
Р
	Assign_95Assign-last_policy_network/last_policy_network/fc0/w8current_policy_network/current_policy_network/fc0/w/read*
use_locking( *
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
М
	Assign_96Assign-last_policy_network/last_policy_network/fc1/b8current_policy_network/current_policy_network/fc1/b/read*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
Р
	Assign_97Assign-last_policy_network/last_policy_network/fc1/w8current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*@
_class6
42loc:@last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
М
	Assign_98Assign-last_policy_network/last_policy_network/out/b8current_policy_network/current_policy_network/out/b/read*
use_locking( *
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:
Р
	Assign_99Assign-last_policy_network/last_policy_network/out/w8current_policy_network/current_policy_network/out/w/read*
use_locking( *
T0*@
_class6
42loc:@last_policy_network/last_policy_network/out/w*
validate_shape(*
_output_shapes

:@
М
group_deps_7NoOp
^Assign_90
^Assign_91
^Assign_92
^Assign_93
^Assign_94
^Assign_95
^Assign_96
^Assign_97
^Assign_98
^Assign_99
ш

Assign_100Assign"best_policy_network/LayerNorm/beta)target_policy_network/LayerNorm/beta/read*
T0*5
_class+
)'loc:@best_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@*
use_locking( 
ы

Assign_101Assign#best_policy_network/LayerNorm/gamma*target_policy_network/LayerNorm/gamma/read*6
_class,
*(loc:@best_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
ю

Assign_102Assign$best_policy_network/LayerNorm_1/beta+target_policy_network/LayerNorm_1/beta/read*
_output_shapes
:@*
use_locking( *
T0*7
_class-
+)loc:@best_policy_network/LayerNorm_1/beta*
validate_shape(
ё

Assign_103Assign%best_policy_network/LayerNorm_1/gamma,target_policy_network/LayerNorm_1/gamma/read*
T0*8
_class.
,*loc:@best_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( 
Л

Assign_104Assign-best_policy_network/best_policy_network/fc0/b6target_policy_network/target_policy_network/fc0/b/read*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
П

Assign_105Assign-best_policy_network/best_policy_network/fc0/w6target_policy_network/target_policy_network/fc0/w/read*
_output_shapes

:@*
use_locking( *
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc0/w*
validate_shape(
Л

Assign_106Assign-best_policy_network/best_policy_network/fc1/b6target_policy_network/target_policy_network/fc1/b/read*
use_locking( *
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
П

Assign_107Assign-best_policy_network/best_policy_network/fc1/w6target_policy_network/target_policy_network/fc1/w/read*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 
Л

Assign_108Assign-best_policy_network/best_policy_network/out/b6target_policy_network/target_policy_network/out/b/read*
T0*@
_class6
42loc:@best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
П

Assign_109Assign-best_policy_network/best_policy_network/out/w6target_policy_network/target_policy_network/out/w/read*@
_class6
42loc:@best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
Ц
group_deps_8NoOp^Assign_100^Assign_101^Assign_102^Assign_103^Assign_104^Assign_105^Assign_106^Assign_107^Assign_108^Assign_109
ъ

Assign_110Assign$target_policy_network/LayerNorm/beta'best_policy_network/LayerNorm/beta/read*
use_locking( *
T0*7
_class-
+)loc:@target_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
э

Assign_111Assign%target_policy_network/LayerNorm/gamma(best_policy_network/LayerNorm/gamma/read*
use_locking( *
T0*8
_class.
,*loc:@target_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@
Ё

Assign_112Assign&target_policy_network/LayerNorm_1/beta)best_policy_network/LayerNorm_1/beta/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*9
_class/
-+loc:@target_policy_network/LayerNorm_1/beta
є

Assign_113Assign'target_policy_network/LayerNorm_1/gamma*best_policy_network/LayerNorm_1/gamma/read*:
_class0
.,loc:@target_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0
П

Assign_114Assign1target_policy_network/target_policy_network/fc0/b2best_policy_network/best_policy_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/b*
validate_shape(
У

Assign_115Assign1target_policy_network/target_policy_network/fc0/w2best_policy_network/best_policy_network/fc0/w/read*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
П

Assign_116Assign1target_policy_network/target_policy_network/fc1/b2best_policy_network/best_policy_network/fc1/b/read*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
У

Assign_117Assign1target_policy_network/target_policy_network/fc1/w2best_policy_network/best_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
П

Assign_118Assign1target_policy_network/target_policy_network/out/b2best_policy_network/best_policy_network/out/b/read*
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
У

Assign_119Assign1target_policy_network/target_policy_network/out/w2best_policy_network/best_policy_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@target_policy_network/target_policy_network/out/w
ь

Assign_120Assign%current_policy_network/LayerNorm/beta'best_policy_network/LayerNorm/beta/read*
use_locking( *
T0*8
_class.
,*loc:@current_policy_network/LayerNorm/beta*
validate_shape(*
_output_shapes
:@
я

Assign_121Assign&current_policy_network/LayerNorm/gamma(best_policy_network/LayerNorm/gamma/read*
T0*9
_class/
-+loc:@current_policy_network/LayerNorm/gamma*
validate_shape(*
_output_shapes
:@*
use_locking( 
Є

Assign_122Assign'current_policy_network/LayerNorm_1/beta)best_policy_network/LayerNorm_1/beta/read*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@current_policy_network/LayerNorm_1/beta*
validate_shape(
ї

Assign_123Assign(current_policy_network/LayerNorm_1/gamma*best_policy_network/LayerNorm_1/gamma/read*
use_locking( *
T0*;
_class1
/-loc:@current_policy_network/LayerNorm_1/gamma*
validate_shape(*
_output_shapes
:@
У

Assign_124Assign3current_policy_network/current_policy_network/fc0/b2best_policy_network/best_policy_network/fc0/b/read*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( 
Ч

Assign_125Assign3current_policy_network/current_policy_network/fc0/w2best_policy_network/best_policy_network/fc0/w/read*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
У

Assign_126Assign3current_policy_network/current_policy_network/fc1/b2best_policy_network/best_policy_network/fc1/b/read*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
Ч

Assign_127Assign3current_policy_network/current_policy_network/fc1/w2best_policy_network/best_policy_network/fc1/w/read*
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 
У

Assign_128Assign3current_policy_network/current_policy_network/out/b2best_policy_network/best_policy_network/out/b/read*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
Ч

Assign_129Assign3current_policy_network/current_policy_network/out/w2best_policy_network/best_policy_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*F
_class<
:8loc:@current_policy_network/current_policy_network/out/w*
validate_shape(
Ш
group_deps_9NoOp^Assign_110^Assign_111^Assign_112^Assign_113^Assign_114^Assign_115^Assign_116^Assign_117^Assign_118^Assign_119^Assign_120^Assign_121^Assign_122^Assign_123^Assign_124^Assign_125^Assign_126^Assign_127^Assign_128^Assign_129
S
average_rewardPlaceholder*
dtype0*
_output_shapes
:*
shape:
f
average_reward_1/tagsConst*!
valueB Baverage_reward_1*
dtype0*
_output_shapes
: 
i
average_reward_1ScalarSummaryaverage_reward_1/tagsaverage_reward*
T0*
_output_shapes
: "ў.Р