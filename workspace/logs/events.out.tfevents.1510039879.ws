       ЃK"	  РQXжAbrain.Event:2э2тьЧ     іПс	ћя№QXжA"п
s
A2S/observationsPlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
n
A2S/actionsPlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
q
A2S/advantagesPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
V
A2S/learning_ratePlaceholder*
dtype0*
_output_shapes
:*
shape:
Y
A2S/last_mean_policyPlaceholder*
dtype0*
_output_shapes
:*
shape:
\
A2S/last_std_dev_policyPlaceholder*
shape:*
dtype0*
_output_shapes
:
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
ѕ
XA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
ч
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
ч
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ц
`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
seed2*
dtype0*
_output_shapes

:@
њ
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes
: 

VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
ў
RA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
ї
7A2S/current_policy_network/current_policy_network/fc0/w
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
ѓ
>A2S/current_policy_network/current_policy_network/fc0/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/wRA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
і
<A2S/current_policy_network/current_policy_network/fc0/w/readIdentity7A2S/current_policy_network/current_policy_network/fc0/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
т
IA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
я
7A2S/current_policy_network/current_policy_network/fc0/b
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
ц
>A2S/current_policy_network/current_policy_network/fc0/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/bIA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
ђ
<A2S/current_policy_network/current_policy_network/fc0/b/readIdentity7A2S/current_policy_network/current_policy_network/fc0/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
г
!A2S/current_policy_network/MatMulMatMulA2S/observations<A2S/current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
И
A2S/current_policy_network/addAdd!A2S/current_policy_network/MatMul<A2S/current_policy_network/current_policy_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
y
A2S/current_policy_network/TanhTanhA2S/current_policy_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
ѕ
XA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
ч
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  П*
dtype0
ч
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxConst*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  ?*
dtype0
ц
`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
seed2
њ
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w

VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
ў
RA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
ї
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
ѓ
>A2S/current_policy_network/current_policy_network/fc1/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/wRA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
і
<A2S/current_policy_network/current_policy_network/fc1/w/readIdentity7A2S/current_policy_network/current_policy_network/fc1/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
т
IA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
я
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
ц
>A2S/current_policy_network/current_policy_network/fc1/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/bIA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
ђ
<A2S/current_policy_network/current_policy_network/fc1/b/readIdentity7A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b
ф
#A2S/current_policy_network/MatMul_1MatMulA2S/current_policy_network/Tanh<A2S/current_policy_network/current_policy_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
М
 A2S/current_policy_network/add_1Add#A2S/current_policy_network/MatMul_1<A2S/current_policy_network/current_policy_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
}
!A2S/current_policy_network/Tanh_1Tanh A2S/current_policy_network/add_1*'
_output_shapes
:џџџџџџџџџ@*
T0
ѕ
XA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
ч
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
ч
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *ЭЬЬ=
ц
`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shape*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
seed2-*
dtype0*
_output_shapes

:@*

seed
њ
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes
: 

VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/sub*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@*
T0
ў
RA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ї
7A2S/current_policy_network/current_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@
ѓ
>A2S/current_policy_network/current_policy_network/out/w/AssignAssign7A2S/current_policy_network/current_policy_network/out/wRA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w
і
<A2S/current_policy_network/current_policy_network/out/w/readIdentity7A2S/current_policy_network/current_policy_network/out/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
т
IA2S/current_policy_network/current_policy_network/out/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
я
7A2S/current_policy_network/current_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:
ц
>A2S/current_policy_network/current_policy_network/out/b/AssignAssign7A2S/current_policy_network/current_policy_network/out/bIA2S/current_policy_network/current_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
ђ
<A2S/current_policy_network/current_policy_network/out/b/readIdentity7A2S/current_policy_network/current_policy_network/out/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
ц
#A2S/current_policy_network/MatMul_2MatMul!A2S/current_policy_network/Tanh_1<A2S/current_policy_network/current_policy_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
М
 A2S/current_policy_network/add_2Add#A2S/current_policy_network/MatMul_2<A2S/current_policy_network/current_policy_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
щ
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"   @   *
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
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
seed2=
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

:@*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
ц
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
ы
1A2S/best_policy_network/best_policy_network/fc0/w
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
л
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
ф
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
ж
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:@
Ю
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
р
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:@
Ъ
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
Ќ
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
s
A2S/best_policy_network/TanhTanhA2S/best_policy_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
щ
RA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB"@   @   
л
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  П*
dtype0
л
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
д
ZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
seed2N
т
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes
: 
є
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/sub*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@*
T0
ц
LA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
ы
1A2S/best_policy_network/best_policy_network/fc1/w
VariableV2*
_output_shapes

:@@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
	container *
shape
:@@*
dtype0
л
8A2S/best_policy_network/best_policy_network/fc1/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/wLA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
ф
6A2S/best_policy_network/best_policy_network/fc1/w/readIdentity1A2S/best_policy_network/best_policy_network/fc1/w*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@*
T0
ж
CA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
valueB@*    
у
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
Ю
8A2S/best_policy_network/best_policy_network/fc1/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/bCA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zeros*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
р
6A2S/best_policy_network/best_policy_network/fc1/b/readIdentity1A2S/best_policy_network/best_policy_network/fc1/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
_output_shapes
:@
и
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/Tanh6A2S/best_policy_network/best_policy_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
А
A2S/best_policy_network/add_1Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/fc1/b/read*'
_output_shapes
:џџџџџџџџџ@*
T0
w
A2S/best_policy_network/Tanh_1TanhA2S/best_policy_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
щ
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"@      *
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

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2_*
dtype0*
_output_shapes

:@
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

:@
ц
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
ы
1A2S/best_policy_network/best_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:@
л
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@
ф
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
ж
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    
у
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:
Ю
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
р
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b
к
 A2S/best_policy_network/MatMul_2MatMulA2S/best_policy_network/Tanh_16A2S/best_policy_network/best_policy_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
А
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_26A2S/best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
щ
RA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
л
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
л
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
д
ZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shape*
seed2o*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w
т
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes
: 
є
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w
ц
LA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
ы
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
л
8A2S/last_policy_network/last_policy_network/fc0/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/wLA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
ф
6A2S/last_policy_network/last_policy_network/fc0/w/readIdentity1A2S/last_policy_network/last_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
ж
CA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
valueB@*    
у
1A2S/last_policy_network/last_policy_network/fc0/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
	container 
Ю
8A2S/last_policy_network/last_policy_network/fc0/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/bCA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b
р
6A2S/last_policy_network/last_policy_network/fc0/b/readIdentity1A2S/last_policy_network/last_policy_network/fc0/b*
_output_shapes
:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b
Ъ
A2S/last_policy_network/MatMulMatMulA2S/observations6A2S/last_policy_network/last_policy_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Ќ
A2S/last_policy_network/addAddA2S/last_policy_network/MatMul6A2S/last_policy_network/last_policy_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
s
A2S/last_policy_network/TanhTanhA2S/last_policy_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
щ
RA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
л
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
л
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
е
ZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
seed2
т
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes
: 
є
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w
ц
LA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
ы
1A2S/last_policy_network/last_policy_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
	container *
shape
:@@
л
8A2S/last_policy_network/last_policy_network/fc1/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/wLA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ф
6A2S/last_policy_network/last_policy_network/fc1/w/readIdentity1A2S/last_policy_network/last_policy_network/fc1/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
ж
CA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1A2S/last_policy_network/last_policy_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
	container *
shape:@
Ю
8A2S/last_policy_network/last_policy_network/fc1/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/bCA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
validate_shape(
р
6A2S/last_policy_network/last_policy_network/fc1/b/readIdentity1A2S/last_policy_network/last_policy_network/fc1/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
_output_shapes
:@
и
 A2S/last_policy_network/MatMul_1MatMulA2S/last_policy_network/Tanh6A2S/last_policy_network/last_policy_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
А
A2S/last_policy_network/add_1Add A2S/last_policy_network/MatMul_16A2S/last_policy_network/last_policy_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
w
A2S/last_policy_network/Tanh_1TanhA2S/last_policy_network/add_1*'
_output_shapes
:џџџџџџџџџ@*
T0
щ
RA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
л
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *ЭЬЬН
л
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
е
ZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
seed2
т
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
є
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
ц
LA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
ы
1A2S/last_policy_network/last_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
	container *
shape
:@
л
8A2S/last_policy_network/last_policy_network/out/w/AssignAssign1A2S/last_policy_network/last_policy_network/out/wLA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
ф
6A2S/last_policy_network/last_policy_network/out/w/readIdentity1A2S/last_policy_network/last_policy_network/out/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
ж
CA2S/last_policy_network/last_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
у
1A2S/last_policy_network/last_policy_network/out/b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
	container *
shape:
Ю
8A2S/last_policy_network/last_policy_network/out/b/AssignAssign1A2S/last_policy_network/last_policy_network/out/bCA2S/last_policy_network/last_policy_network/out/b/Initializer/zeros*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
р
6A2S/last_policy_network/last_policy_network/out/b/readIdentity1A2S/last_policy_network/last_policy_network/out/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
_output_shapes
:
к
 A2S/last_policy_network/MatMul_2MatMulA2S/last_policy_network/Tanh_16A2S/last_policy_network/last_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
A2S/last_policy_network/add_2Add A2S/last_policy_network/MatMul_26A2S/last_policy_network/last_policy_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
ё
VA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
у
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
у
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
с
^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shape*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
seed2Ё*
dtype0*
_output_shapes

:@*

seed*
T0
ђ
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w

TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w
і
PA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
ѓ
5A2S/current_value_network/current_value_network/fc0/w
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
ы
<A2S/current_value_network/current_value_network/fc0/w/AssignAssign5A2S/current_value_network/current_value_network/fc0/wPA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
№
:A2S/current_value_network/current_value_network/fc0/w/readIdentity5A2S/current_value_network/current_value_network/fc0/w*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
о
GA2S/current_value_network/current_value_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ы
5A2S/current_value_network/current_value_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape:@
о
<A2S/current_value_network/current_value_network/fc0/b/AssignAssign5A2S/current_value_network/current_value_network/fc0/bGA2S/current_value_network/current_value_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
ь
:A2S/current_value_network/current_value_network/fc0/b/readIdentity5A2S/current_value_network/current_value_network/fc0/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
а
 A2S/current_value_network/MatMulMatMulA2S/observations:A2S/current_value_network/current_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Д
A2S/current_value_network/addAdd A2S/current_value_network/MatMul:A2S/current_value_network/current_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
w
A2S/current_value_network/TanhTanhA2S/current_value_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
ё
VA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
у
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  П*
dtype0
у
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  ?
с
^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shape*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
seed2В*
dtype0*
_output_shapes

:@@
ђ
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w

TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/sub*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0
і
PA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
ѓ
5A2S/current_value_network/current_value_network/fc1/w
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container 
ы
<A2S/current_value_network/current_value_network/fc1/w/AssignAssign5A2S/current_value_network/current_value_network/fc1/wPA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
№
:A2S/current_value_network/current_value_network/fc1/w/readIdentity5A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
о
GA2S/current_value_network/current_value_network/fc1/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ы
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
о
<A2S/current_value_network/current_value_network/fc1/b/AssignAssign5A2S/current_value_network/current_value_network/fc1/bGA2S/current_value_network/current_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
ь
:A2S/current_value_network/current_value_network/fc1/b/readIdentity5A2S/current_value_network/current_value_network/fc1/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@
р
"A2S/current_value_network/MatMul_1MatMulA2S/current_value_network/Tanh:A2S/current_value_network/current_value_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
И
A2S/current_value_network/add_1Add"A2S/current_value_network/MatMul_1:A2S/current_value_network/current_value_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
{
 A2S/current_value_network/Tanh_1TanhA2S/current_value_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
ё
VA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
у
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
у
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
с
^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shape*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
seed2У*
dtype0*
_output_shapes

:@*

seed*
T0
ђ
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w

TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
і
PA2S/current_value_network/current_value_network/out/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
ѓ
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
ы
<A2S/current_value_network/current_value_network/out/w/AssignAssign5A2S/current_value_network/current_value_network/out/wPA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
№
:A2S/current_value_network/current_value_network/out/w/readIdentity5A2S/current_value_network/current_value_network/out/w*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
о
GA2S/current_value_network/current_value_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    
ы
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
о
<A2S/current_value_network/current_value_network/out/b/AssignAssign5A2S/current_value_network/current_value_network/out/bGA2S/current_value_network/current_value_network/out/b/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
ь
:A2S/current_value_network/current_value_network/out/b/readIdentity5A2S/current_value_network/current_value_network/out/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:
т
"A2S/current_value_network/MatMul_2MatMul A2S/current_value_network/Tanh_1:A2S/current_value_network/current_value_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
И
A2S/current_value_network/add_2Add"A2S/current_value_network/MatMul_2:A2S/current_value_network/current_value_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
х
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"   @   *
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
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB
 *  ?
Я
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2г
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

:@
о
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@
ч
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container 
г
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
о
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
в
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*
dtype0*
_output_shapes
:@*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB@*    
п
/A2S/best_value_network/best_value_network/fc0/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
	container *
shape:@
Ц
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
к
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:@
Ч
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Ј
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
q
A2S/best_value_network/TanhTanhA2S/best_value_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
х
PA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB"@   @   *
dtype0
з
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
з
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Я
XA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shape*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
seed2ф*
dtype0*
_output_shapes

:@@
к
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes
: 
ь
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
о
JA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@*
T0
ч
/A2S/best_value_network/best_value_network/fc1/w
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
	container 
г
6A2S/best_value_network/best_value_network/fc1/w/AssignAssign/A2S/best_value_network/best_value_network/fc1/wJA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
о
4A2S/best_value_network/best_value_network/fc1/w/readIdentity/A2S/best_value_network/best_value_network/fc1/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
в
AA2S/best_value_network/best_value_network/fc1/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
п
/A2S/best_value_network/best_value_network/fc1/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
	container 
Ц
6A2S/best_value_network/best_value_network/fc1/b/AssignAssign/A2S/best_value_network/best_value_network/fc1/bAA2S/best_value_network/best_value_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(
к
4A2S/best_value_network/best_value_network/fc1/b/readIdentity/A2S/best_value_network/best_value_network/fc1/b*
_output_shapes
:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b
д
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/Tanh4A2S/best_value_network/best_value_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Ќ
A2S/best_value_network/add_1AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
u
A2S/best_value_network/Tanh_1TanhA2S/best_value_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
х
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"@      *
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

:@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2ѕ
к
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
ь
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
о
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
ч
/A2S/best_value_network/best_value_network/out/w
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
г
6A2S/best_value_network/best_value_network/out/w/AssignAssign/A2S/best_value_network/best_value_network/out/wJA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@
о
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@*
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
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ц
6A2S/best_value_network/best_value_network/out/b/AssignAssign/A2S/best_value_network/best_value_network/out/bAA2S/best_value_network/best_value_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(
к
4A2S/best_value_network/best_value_network/out/b/readIdentity/A2S/best_value_network/best_value_network/out/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
_output_shapes
:
ж
A2S/best_value_network/MatMul_2MatMulA2S/best_value_network/Tanh_14A2S/best_value_network/best_value_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ќ
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_24A2S/best_value_network/best_value_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
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
Е
A2S/strided_sliceStridedSlice A2S/current_policy_network/add_2A2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ
`
A2S/SqueezeSqueezeA2S/strided_slice*
_output_shapes
:*
squeeze_dims
 *
T0
b
A2S/Reshape/shapeConst*
_output_shapes
:*
valueB"џџџџ   *
dtype0
v
A2S/ReshapeReshapeA2S/SqueezeA2S/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
К
A2S/strided_slice_1StridedSliceA2S/best_policy_network/add_2A2S/strided_slice_1/stackA2S/strided_slice_1/stack_1A2S/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ
d
A2S/Squeeze_1SqueezeA2S/strided_slice_1*
squeeze_dims
 *
T0*
_output_shapes
:
d
A2S/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   
|
A2S/Reshape_1ReshapeA2S/Squeeze_1A2S/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
A2S/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
К
A2S/strided_slice_2StridedSliceA2S/last_policy_network/add_2A2S/strided_slice_2/stackA2S/strided_slice_2/stack_1A2S/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ
d
A2S/Squeeze_2SqueezeA2S/strided_slice_2*
squeeze_dims
 *
T0*
_output_shapes
:
d
A2S/Reshape_2/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
|
A2S/Reshape_2ReshapeA2S/Squeeze_2A2S/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
N
	A2S/ConstConst*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
P
A2S/Const_1Const*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
P
A2S/Const_2Const*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Y
A2S/Normal/locIdentityA2S/Reshape*'
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ
L
A2S/Normal_1/scaleIdentityA2S/Const_1*
_output_shapes
: *
T0
]
A2S/Normal_2/locIdentityA2S/Reshape_2*
T0*'
_output_shapes
:џџџџџџџџџ
L
A2S/Normal_2/scaleIdentityA2S/Const_2*
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
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   @
q
,A2S/KullbackLeibler/kl_normal_normal/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
h
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal/scale*
T0*
_output_shapes
: 
l
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal_2/scale*
T0*
_output_shapes
: 
Д
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 

(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal/locA2S/Normal_2/loc*
T0*'
_output_shapes
:џџџџџџџџџ

-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:џџџџџџџџџ
­
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 
Ф
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:џџџџџџџџџ
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
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
_output_shapes
: *
T0
Н
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_3Const*
valueB"       *
dtype0*
_output_shapes
:

A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_3*
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
s
%A2S/Normal_3/batch_shape_tensor/ShapeShapeA2S/Normal/loc*
T0*
out_type0*
_output_shapes
:
j
'A2S/Normal_3/batch_shape_tensor/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Г
-A2S/Normal_3/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_3/batch_shape_tensor/Shape'A2S/Normal_3/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
]
A2S/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
Q
A2S/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ

A2S/concatConcatV2A2S/concat/values_0-A2S/Normal_3/batch_shape_tensor/BroadcastArgsA2S/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
[
A2S/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
]
A2S/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
А
&A2S/random_normal/RandomStandardNormalRandomStandardNormal
A2S/concat*
T0*
dtype0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
seed2Л*

seed

A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
r
A2S/mulMulA2S/random_normalA2S/Normal/scale*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
]
A2S/addAddA2S/mulA2S/Normal/loc*
T0*+
_output_shapes
:џџџџџџџџџ
h
A2S/Reshape_3/shapeConst*
_output_shapes
:*!
valueB"џџџџ      *
dtype0
z
A2S/Reshape_3ReshapeA2S/addA2S/Reshape_3/shape*+
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
S
A2S/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 

A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ*

Tidx0
с
NA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
г
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
г
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  ?
Щ
VA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
seed2Ч*
dtype0*
_output_shapes

:@*

seed
в
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes
: 
ф
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
ж
HA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@*
T0
у
-A2S/current_q_network/current_q_network/fc0/w
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
Ы
4A2S/current_q_network/current_q_network/fc0/w/AssignAssign-A2S/current_q_network/current_q_network/fc0/wHA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
и
2A2S/current_q_network/current_q_network/fc0/w/readIdentity-A2S/current_q_network/current_q_network/fc0/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
Ю
?A2S/current_q_network/current_q_network/fc0/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
л
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
О
4A2S/current_q_network/current_q_network/fc0/b/AssignAssign-A2S/current_q_network/current_q_network/fc0/b?A2S/current_q_network/current_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
д
2A2S/current_q_network/current_q_network/fc0/b/readIdentity-A2S/current_q_network/current_q_network/fc0/b*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@
Р
A2S/current_q_network/MatMulMatMulA2S/concat_12A2S/current_q_network/current_q_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
Є
A2S/current_q_network/addAddA2S/current_q_network/MatMul2A2S/current_q_network/current_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
o
A2S/current_q_network/TanhTanhA2S/current_q_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
с
NA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
г
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
г
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Щ
VA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
seed2и
в
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes
: 
ф
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
ж
HA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
у
-A2S/current_q_network/current_q_network/fc1/w
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
Ы
4A2S/current_q_network/current_q_network/fc1/w/AssignAssign-A2S/current_q_network/current_q_network/fc1/wHA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
и
2A2S/current_q_network/current_q_network/fc1/w/readIdentity-A2S/current_q_network/current_q_network/fc1/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
Ю
?A2S/current_q_network/current_q_network/fc1/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
л
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
О
4A2S/current_q_network/current_q_network/fc1/b/AssignAssign-A2S/current_q_network/current_q_network/fc1/b?A2S/current_q_network/current_q_network/fc1/b/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
д
2A2S/current_q_network/current_q_network/fc1/b/readIdentity-A2S/current_q_network/current_q_network/fc1/b*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@*
T0
а
A2S/current_q_network/MatMul_1MatMulA2S/current_q_network/Tanh2A2S/current_q_network/current_q_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
Ј
A2S/current_q_network/add_1AddA2S/current_q_network/MatMul_12A2S/current_q_network/current_q_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
s
A2S/current_q_network/Tanh_1TanhA2S/current_q_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
с
NA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
г
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/minConst*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *ЭЬЬН*
dtype0
г
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Щ
VA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
seed2щ
в
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes
: 
ф
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w
ж
HA2S/current_q_network/current_q_network/out/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
у
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
Ы
4A2S/current_q_network/current_q_network/out/w/AssignAssign-A2S/current_q_network/current_q_network/out/wHA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
и
2A2S/current_q_network/current_q_network/out/w/readIdentity-A2S/current_q_network/current_q_network/out/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
Ю
?A2S/current_q_network/current_q_network/out/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
л
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
О
4A2S/current_q_network/current_q_network/out/b/AssignAssign-A2S/current_q_network/current_q_network/out/b?A2S/current_q_network/current_q_network/out/b/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
д
2A2S/current_q_network/current_q_network/out/b/readIdentity-A2S/current_q_network/current_q_network/out/b*
_output_shapes
:*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b
в
A2S/current_q_network/MatMul_2MatMulA2S/current_q_network/Tanh_12A2S/current_q_network/current_q_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ј
A2S/current_q_network/add_2AddA2S/current_q_network/MatMul_22A2S/current_q_network/current_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
е
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"   @   *
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

:@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2љ
К
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
Ь
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
О
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@*
T0
з
'A2S/best_q_network/best_q_network/fc0/w
VariableV2*
_output_shapes

:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container *
shape
:@*
dtype0
Г
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
Ц
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
Т
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
Я
'A2S/best_q_network/best_q_network/fc0/b
VariableV2*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
	container *
shape:@*
dtype0
І
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
Т
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:@
З
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 

A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*'
_output_shapes
:џџџџџџџџџ@*
T0
i
A2S/best_q_network/TanhTanhA2S/best_q_network/add*'
_output_shapes
:џџџџџџџџџ@*
T0
е
HA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
Ч
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
Ч
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
З
PA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes

:@@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w
К
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w
Ь
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
О
BA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@*
T0
з
'A2S/best_q_network/best_q_network/fc1/w
VariableV2*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
Г
.A2S/best_q_network/best_q_network/fc1/w/AssignAssign'A2S/best_q_network/best_q_network/fc1/wBA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
Ц
,A2S/best_q_network/best_q_network/fc1/w/readIdentity'A2S/best_q_network/best_q_network/fc1/w*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@*
T0
Т
9A2S/best_q_network/best_q_network/fc1/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
Я
'A2S/best_q_network/best_q_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
	container *
shape:@
І
.A2S/best_q_network/best_q_network/fc1/b/AssignAssign'A2S/best_q_network/best_q_network/fc1/b9A2S/best_q_network/best_q_network/fc1/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
Т
,A2S/best_q_network/best_q_network/fc1/b/readIdentity'A2S/best_q_network/best_q_network/fc1/b*
_output_shapes
:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b
Ф
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/Tanh,A2S/best_q_network/best_q_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0

A2S/best_q_network/add_1AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
m
A2S/best_q_network/Tanh_1TanhA2S/best_q_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
е
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"@      *
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
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
З
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
seed2
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

:@
О
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@*
T0
з
'A2S/best_q_network/best_q_network/out/w
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w
Г
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
Ц
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b
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
Ц
A2S/best_q_network/MatMul_2MatMulA2S/best_q_network/Tanh_1,A2S/best_q_network/best_q_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_2,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
{
%A2S/Normal_4/log_prob/standardize/subSubA2S/actionsA2S/Normal/loc*
T0*'
_output_shapes
:џџџџџџџџџ

)A2S/Normal_4/log_prob/standardize/truedivRealDiv%A2S/Normal_4/log_prob/standardize/subA2S/Normal/scale*
T0*'
_output_shapes
:џџџџџџџџџ

A2S/Normal_4/log_prob/SquareSquare)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
`
A2S/Normal_4/log_prob/mul/xConst*
valueB
 *   П*
dtype0*
_output_shapes
: 

A2S/Normal_4/log_prob/mulMulA2S/Normal_4/log_prob/mul/xA2S/Normal_4/log_prob/Square*'
_output_shapes
:џџџџџџџџџ*
T0
S
A2S/Normal_4/log_prob/LogLogA2S/Normal/scale*
T0*
_output_shapes
: 
`
A2S/Normal_4/log_prob/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 
y
A2S/Normal_4/log_prob/addAddA2S/Normal_4/log_prob/add/xA2S/Normal_4/log_prob/Log*
_output_shapes
: *
T0

A2S/Normal_4/log_prob/subSubA2S/Normal_4/log_prob/mulA2S/Normal_4/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
[
A2S/NegNegA2S/Normal_4/log_prob/sub*
T0*'
_output_shapes
:џџџџџџџџџ
[
	A2S/mul_1MulA2S/NegA2S/advantages*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
h

A2S/Mean_1Mean	A2S/mul_1A2S/Const_4*

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
A2S/Mean_1*
_output_shapes
: *
T0

A2S/SquaredDifferenceSquaredDifferenceA2S/current_value_network/add_2A2S/returns*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_5Const*
_output_shapes
:*
valueB"       *
dtype0
t

A2S/Mean_2MeanA2S/SquaredDifferenceA2S/Const_5*
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
A2S/Mean_2*
T0*
_output_shapes
: 

A2S/SquaredDifference_1SquaredDifferenceA2S/current_q_network/add_2A2S/returns*'
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

A2S/Mean_3MeanA2S/SquaredDifference_1A2S/Const_6*
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
+A2S/gradients/A2S/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ј
%A2S/gradients/A2S/Mean_1_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
#A2S/gradients/A2S/Mean_1_grad/ShapeShape	A2S/mul_1*
out_type0*
_output_shapes
:*
T0
К
"A2S/gradients/A2S/Mean_1_grad/TileTile%A2S/gradients/A2S/Mean_1_grad/Reshape#A2S/gradients/A2S/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
Д
"A2S/gradients/A2S/Mean_1_grad/ProdProd%A2S/gradients/A2S/Mean_1_grad/Shape_1#A2S/gradients/A2S/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
o
%A2S/gradients/A2S/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
И
$A2S/gradients/A2S/Mean_1_grad/Prod_1Prod%A2S/gradients/A2S/Mean_1_grad/Shape_2%A2S/gradients/A2S/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
'A2S/gradients/A2S/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
%A2S/gradients/A2S/Mean_1_grad/MaximumMaximum$A2S/gradients/A2S/Mean_1_grad/Prod_1'A2S/gradients/A2S/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

&A2S/gradients/A2S/Mean_1_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_1_grad/Prod%A2S/gradients/A2S/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

"A2S/gradients/A2S/Mean_1_grad/CastCast&A2S/gradients/A2S/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Њ
%A2S/gradients/A2S/Mean_1_grad/truedivRealDiv"A2S/gradients/A2S/Mean_1_grad/Tile"A2S/gradients/A2S/Mean_1_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
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
в
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_1_grad/truedivA2S/advantages*
T0*'
_output_shapes
:џџџџџџџџџ
Н
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Е
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_1_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
У
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Л
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1

5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@A2S/gradients/A2S/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

7A2S/gradients/A2S/mul_1_grad/tuple/control_dependency_1Identity&A2S/gradients/A2S/mul_1_grad/Reshape_1.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*9
_class/
-+loc:@A2S/gradients/A2S/mul_1_grad/Reshape_1

A2S/gradients/A2S/Neg_grad/NegNeg5A2S/gradients/A2S/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ShapeShapeA2S/Normal_4/log_prob/mul*
out_type0*
_output_shapes
:*
T0
w
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

BA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
л
0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
х
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
п
2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
и
6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
=A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1
Т
EA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
З
GA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1*
_output_shapes
: 
u
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 

4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1ShapeA2S/Normal_4/log_prob/Square*
T0*
out_type0*
_output_shapes
:

BA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_4/log_prob/Square*
T0*'
_output_shapes
:џџџџџџџџџ
э
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
д
4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Я
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1MulA2S/Normal_4/log_prob/mul/xEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Е
=A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1
Б
EA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape
Ш
GA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1
Ф
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ю
3A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/x)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
ь
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_4/log_prob/standardize/sub*
out_type0*
_output_shapes
:*
T0

DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
В
RA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1A2S/Normal/scale*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
 
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_4/log_prob/standardize/sub*'
_output_shapes
:џџџџџџџџџ*
T0
з
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegA2S/Normal/scale*
T0*'
_output_shapes
:џџџџџџџџџ
н
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal/scale*'
_output_shapes
:џџџџџџџџџ*
T0
ј
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
х
MA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1

UA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ї
WA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 

>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
T0*
out_type0*
_output_shapes
:

@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal/loc*
_output_shapes
:*
T0*
out_type0
І
NA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Њ
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ў
>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
І
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:

BA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
й
IA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1
ђ
QA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ј
SA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1
x
$A2S/gradients/A2S/Reshape_grad/ShapeShapeA2S/Squeeze*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
н
&A2S/gradients/A2S/Reshape_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1$A2S/gradients/A2S/Reshape_grad/Shape*
Tshape0*
_output_shapes
:*
T0
u
$A2S/gradients/A2S/Squeeze_grad/ShapeShapeA2S/strided_slice*
T0*
out_type0*
_output_shapes
:
П
&A2S/gradients/A2S/Squeeze_grad/ReshapeReshape&A2S/gradients/A2S/Reshape_grad/Reshape$A2S/gradients/A2S/Squeeze_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

*A2S/gradients/A2S/strided_slice_grad/ShapeShape A2S/current_policy_network/add_2*
T0*
out_type0*
_output_shapes
:

5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradStridedSliceGrad*A2S/gradients/A2S/strided_slice_grad/ShapeA2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2&A2S/gradients/A2S/Squeeze_grad/Reshape*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask

9A2S/gradients/A2S/current_policy_network/add_2_grad/ShapeShape#A2S/current_policy_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

IA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7A2S/gradients/A2S/current_policy_network/add_2_grad/SumSum5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradIA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
њ
;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_2_grad/Sum9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1Sum5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradKA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ѓ
=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
Ъ
DA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1
о
LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
з
NA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
Ћ
=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 

?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1MatMul!A2S/current_policy_network/Tanh_1LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
б
GA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1
ш
OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
х
QA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
я
=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradTanhGrad!A2S/current_policy_network/Tanh_1OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

9A2S/gradients/A2S/current_policy_network/add_1_grad/ShapeShape#A2S/current_policy_network/MatMul_1*
out_type0*
_output_shapes
:*
T0

;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@

IA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7A2S/gradients/A2S/current_policy_network/add_1_grad/SumSum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
њ
;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_1_grad/Sum9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0

9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1Sum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradKA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ѓ
=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Ъ
DA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1
о
LA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape
з
NA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1
Ћ
=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc1/w/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 

?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/current_policy_network/TanhLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
б
GA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1
ш
OA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
х
QA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:@@*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1
ы
;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradTanhGradA2S/current_policy_network/TanhOA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ@*
T0

7A2S/gradients/A2S/current_policy_network/add_grad/ShapeShape!A2S/current_policy_network/MatMul*
_output_shapes
:*
T0*
out_type0

9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@

GA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients/A2S/current_policy_network/add_grad/Shape9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5A2S/gradients/A2S/current_policy_network/add_grad/SumSum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradGA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
є
9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeReshape5A2S/gradients/A2S/current_policy_network/add_grad/Sum7A2S/gradients/A2S/current_policy_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

7A2S/gradients/A2S/current_policy_network/add_grad/Sum_1Sum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1Reshape7A2S/gradients/A2S/current_policy_network/add_grad/Sum_19A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Ф
BA2S/gradients/A2S/current_policy_network/add_grad/tuple/group_depsNoOp:^A2S/gradients/A2S/current_policy_network/add_grad/Reshape<^A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1
ж
JA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependencyIdentity9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeC^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
Я
LA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1Identity;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1C^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1*
_output_shapes
:@
Ї
;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulMatMulJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
є
=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Ы
EA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul>^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1
р
MA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulF^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul
н
OA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1F^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:@
Ў
A2S/beta1_power/initial_valueConst*
valueB
 *fff?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
П
A2S/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
ц
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b

A2S/beta1_power/readIdentityA2S/beta1_power*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
Ў
A2S/beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wО?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0
П
A2S/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape: 
ц
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 

A2S/beta2_power/readIdentityA2S/beta2_power*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
ѓ
RA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    

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

GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w

EA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
ѕ
TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@

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

IA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@

GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
ы
RA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    
ј
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

GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@

EA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@*
T0
э
TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
њ
BA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container 

IA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b

GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
ѓ
RA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    

@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam
VariableV2*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 

GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@

EA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
ѕ
TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@

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

IA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@

GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
ы
RA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ј
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

GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@

EA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
э
TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
њ
BA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container *
shape:@

IA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@

GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
ѓ
RA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@

@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container 

GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamRA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@

EA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ѕ
TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@

BA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@

IA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@

GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ы
RA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    
ј
@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:

GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamRA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(

EA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
э
TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    
њ
BA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1
VariableV2*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0

IA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(

GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
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
ћ
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/w@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonOA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
use_nesterov( 
є
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/b@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
use_nesterov( *
_output_shapes
:@*
use_locking( 
§
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/w@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:@@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
use_nesterov( 
і
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/b@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
use_nesterov( 
§
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/w@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w
і
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/b@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
use_nesterov( *
_output_shapes
:

A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
Ю
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
 
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*
_output_shapes
: *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
в
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 
Ў
A2S/AdamNoOpR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
X
A2S/gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
-A2S/gradients_1/A2S/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ў
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
Р
$A2S/gradients_1/A2S/Mean_2_grad/TileTile'A2S/gradients_1/A2S/Mean_2_grad/Reshape%A2S/gradients_1/A2S/Mean_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
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
К
$A2S/gradients_1/A2S/Mean_2_grad/ProdProd'A2S/gradients_1/A2S/Mean_2_grad/Shape_1%A2S/gradients_1/A2S/Mean_2_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
q
'A2S/gradients_1/A2S/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
О
&A2S/gradients_1/A2S/Mean_2_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_2_grad/Shape_2'A2S/gradients_1/A2S/Mean_2_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
k
)A2S/gradients_1/A2S/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'A2S/gradients_1/A2S/Mean_2_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_2_grad/Prod_1)A2S/gradients_1/A2S/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
Є
(A2S/gradients_1/A2S/Mean_2_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_2_grad/Prod'A2S/gradients_1/A2S/Mean_2_grad/Maximum*
_output_shapes
: *
T0

$A2S/gradients_1/A2S/Mean_2_grad/CastCast(A2S/gradients_1/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_1/A2S/Mean_2_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_2_grad/Tile$A2S/gradients_1/A2S/Mean_2_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

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
ќ
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
 
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
У
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_2_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
П
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/current_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_2_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1Mul.A2S/gradients_1/A2S/SquaredDifference_grad/mul.A2S/gradients_1/A2S/SquaredDifference_grad/sub*'
_output_shapes
:џџџџџџџџџ*
T0
щ
.A2S/gradients_1/A2S/SquaredDifference_grad/SumSum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
п
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
э
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
х
4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1Reshape0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_12A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

.A2S/gradients_1/A2S/SquaredDifference_grad/NegNeg4A2S/gradients_1/A2S/SquaredDifference_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
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

:A2S/gradients_1/A2S/current_value_network/add_2_grad/ShapeShape"A2S/current_value_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

JA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

8A2S/gradients_1/A2S/current_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
§
<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyLA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
і
>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Э
EA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1
т
MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape
л
OA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1
Ћ
>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(

@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1MatMul A2S/current_value_network/Tanh_1MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
д
HA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1
ь
PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul
щ
RA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
№
>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradTanhGrad A2S/current_value_network/Tanh_1PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

:A2S/gradients_1/A2S/current_value_network/add_1_grad/ShapeShape"A2S/current_value_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@

JA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

8A2S/gradients_1/A2S/current_value_network/add_1_grad/SumSum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
§
<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1Sum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradLA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
і
>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Э
EA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1
т
MA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
л
OA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1*
_output_shapes
:@
Ћ
>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc1/w/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 

@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1MatMulA2S/current_value_network/TanhMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
д
HA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1
ь
PA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@*
T0
щ
RA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:@@*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1
ь
<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradTanhGradA2S/current_value_network/TanhPA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

8A2S/gradients_1/A2S/current_value_network/add_grad/ShapeShape A2S/current_value_network/MatMul*
T0*
out_type0*
_output_shapes
:

:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:

HA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs8A2S/gradients_1/A2S/current_value_network/add_grad/Shape:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

6A2S/gradients_1/A2S/current_value_network/add_grad/SumSum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradHA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ї
:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeReshape6A2S/gradients_1/A2S/current_value_network/add_grad/Sum8A2S/gradients_1/A2S/current_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1Sum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
№
<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1Reshape8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Ч
CA2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_depsNoOp;^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape=^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1
к
KA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependencyIdentity:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeD^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
г
MA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1Identity<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1D^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1*
_output_shapes
:@
Ї
<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulMatMulKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
і
>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Ю
FA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul?^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1
ф
NA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulG^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
с
PA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1G^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1*
_output_shapes

:@
Ў
A2S/beta1_power_1/initial_valueConst*
valueB
 *fff?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0*
_output_shapes
: 
П
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
ъ
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
 
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
Ў
A2S/beta2_power_1/initial_valueConst*
valueB
 *wО?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0*
_output_shapes
: 
П
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
ъ
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
 
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
я
PA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zerosConst*
_output_shapes

:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0
ќ
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
§
EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/w/AdamPA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@

CA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
ё
RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
ў
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

GA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@

EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
ч
PA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
є
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
љ
EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/b/AdamPA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
ў
CA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
щ
RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
і
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
џ
GA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(

EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
я
PA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
ќ
>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam
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
§
EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/w/AdamPA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w

CA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
ё
RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
ў
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

GA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w

EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@*
T0
ч
PA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
є
>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@
љ
EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/b/AdamPA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
ў
CA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
щ
RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
і
@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
џ
GA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@

EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@
я
PA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ќ
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
§
EA2S/A2S/current_value_network/current_value_network/out/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/w/AdamPA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(

CA2S/A2S/current_value_network/current_value_network/out/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/w/Adam*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@*
T0
ё
RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ў
@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1
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

GA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w

EA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
ч
PA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
є
>A2S/A2S/current_value_network/current_value_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container *
shape:
љ
EA2S/A2S/current_value_network/current_value_network/out/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/b/AdamPA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
ў
CA2S/A2S/current_value_network/current_value_network/out/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/b/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:
щ
RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
і
@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b
џ
GA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:

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
ў
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/w>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonPA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
use_nesterov( *
_output_shapes

:@*
use_locking( 
ї
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/b>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b

QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/w>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
use_nesterov( *
_output_shapes

:@@
љ
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/b>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
use_nesterov( 

QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/w>A2S/A2S/current_value_network/current_value_network/out/w/Adam@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
use_nesterov( *
_output_shapes

:@
љ
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/b>A2S/A2S/current_value_network/current_value_network/out/b/Adam@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
use_nesterov( 
Ђ
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
в
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
Є
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
ж
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
Д

A2S/Adam_1NoOpR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
X
A2S/gradients_2/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
A2S/gradients_2/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
k
A2S/gradients_2/FillFillA2S/gradients_2/ShapeA2S/gradients_2/Const*
_output_shapes
: *
T0
~
-A2S/gradients_2/A2S/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ў
'A2S/gradients_2/A2S/Mean_3_grad/ReshapeReshapeA2S/gradients_2/Fill-A2S/gradients_2/A2S/Mean_3_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
|
%A2S/gradients_2/A2S/Mean_3_grad/ShapeShapeA2S/SquaredDifference_1*
T0*
out_type0*
_output_shapes
:
Р
$A2S/gradients_2/A2S/Mean_3_grad/TileTile'A2S/gradients_2/A2S/Mean_3_grad/Reshape%A2S/gradients_2/A2S/Mean_3_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
%A2S/gradients_2/A2S/Mean_3_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
К
$A2S/gradients_2/A2S/Mean_3_grad/ProdProd'A2S/gradients_2/A2S/Mean_3_grad/Shape_1%A2S/gradients_2/A2S/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
q
'A2S/gradients_2/A2S/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
О
&A2S/gradients_2/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_3_grad/Shape_2'A2S/gradients_2/A2S/Mean_3_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
k
)A2S/gradients_2/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'A2S/gradients_2/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_3_grad/Prod_1)A2S/gradients_2/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
Є
(A2S/gradients_2/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_3_grad/Prod'A2S/gradients_2/A2S/Mean_3_grad/Maximum*
T0*
_output_shapes
: 

$A2S/gradients_2/A2S/Mean_3_grad/CastCast(A2S/gradients_2/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_2/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_3_grad/Tile$A2S/gradients_2/A2S/Mean_3_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

2A2S/gradients_2/A2S/SquaredDifference_1_grad/ShapeShapeA2S/current_q_network/add_2*
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

BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ђ
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
Ч
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Н
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/current_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_3_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Я
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*'
_output_shapes
:џџџџџџџџџ*
T0
я
0A2S/gradients_2/A2S/SquaredDifference_1_grad/SumSum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
х
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
ѓ
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
EA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyIdentity4A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*G
_class=
;9loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape
М
GA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependency_1Identity0A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg>^A2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*C
_class9
75loc:@A2S/gradients_2/A2S/SquaredDifference_1_grad/Neg

6A2S/gradients_2/A2S/current_q_network/add_2_grad/ShapeShapeA2S/current_q_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

FA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

4A2S/gradients_2/A2S/current_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyFA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ё
8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyHA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ъ
:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
С
AA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1
в
IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ы
KA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1

:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/out/w/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
ў
<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1MatMulA2S/current_q_network/Tanh_1IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
Ш
DA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1
м
LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul
й
NA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*
_output_shapes

:@*
T0*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1
ф
:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradTanhGradA2S/current_q_network/Tanh_1LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

6A2S/gradients_2/A2S/current_q_network/add_1_grad/ShapeShapeA2S/current_q_network/MatMul_1*
_output_shapes
:*
T0*
out_type0

8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:@*
dtype0

FA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
4A2S/gradients_2/A2S/current_q_network/add_1_grad/SumSum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ё
8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_1Sum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradHA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ъ
:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
С
AA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1
в
IA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@*
T0
Ы
KA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1

:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
ќ
<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1MatMulA2S/current_q_network/TanhIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
Ш
DA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1
м
LA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
й
NA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
р
8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradTanhGradA2S/current_q_network/TanhLA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

4A2S/gradients_2/A2S/current_q_network/add_grad/ShapeShapeA2S/current_q_network/MatMul*
out_type0*
_output_shapes
:*
T0

6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@

DA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients_2/A2S/current_q_network/add_grad/Shape6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
2A2S/gradients_2/A2S/current_q_network/add_grad/SumSum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradDA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
6A2S/gradients_2/A2S/current_q_network/add_grad/ReshapeReshape2A2S/gradients_2/A2S/current_q_network/add_grad/Sum4A2S/gradients_2/A2S/current_q_network/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0
§
4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_1Sum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ф
8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1Reshape4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_16A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Л
?A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_depsNoOp7^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape9^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1
Ъ
GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients_2/A2S/current_q_network/add_grad/Reshape@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
У
IA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1*
_output_shapes
:@

8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulMatMulGA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ъ
:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
Т
BA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul;^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1
д
JA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulC^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
б
LA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1C^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1*
_output_shapes

:@
І
A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
З
A2S/beta1_power_2
VariableV2*
_output_shapes
: *
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: *
dtype0
т
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 

A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
І
A2S/beta2_power_2/initial_valueConst*
valueB
 *wО?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
З
A2S/beta2_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: 
т
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 

A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
п
HA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zerosConst*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    *
dtype0
ь
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
н
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/w/AdamHA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
ъ
;A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam*
_output_shapes

:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w
с
JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    
ю
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
у
?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
ю
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
з
HA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ф
6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
й
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/b/AdamHA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
ц
;A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@
й
JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zerosConst*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0
ц
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
п
?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ъ
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
п
HA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
ь
6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container *
shape
:@@
н
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/w/AdamHA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w
ъ
;A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
с
JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
ю
8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1
VariableV2*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
у
?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ю
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
з
HA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ф
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
й
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/b/AdamHA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
ц
;A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@*
T0
й
JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ц
8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
п
?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
ъ
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
п
HA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ь
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
н
=A2S/A2S/current_q_network/current_q_network/out/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/w/AdamHA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
ъ
;A2S/A2S/current_q_network/current_q_network/out/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
с
JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ю
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
у
?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(
ю
=A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
з
HA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ф
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
й
=A2S/A2S/current_q_network/current_q_network/out/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/b/AdamHA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
ц
;A2S/A2S/current_q_network/current_q_network/out/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/b/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:
й
JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    
ц
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
п
?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b
ъ
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
в
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/w6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonLA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
use_nesterov( *
_output_shapes

:@
Ы
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/b6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
use_nesterov( *
_output_shapes
:@*
use_locking( 
д
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/w6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
Э
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/b6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
д
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/w6A2S/A2S/current_q_network/current_q_network/out/w/Adam8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
use_nesterov( *
_output_shapes

:@*
use_locking( 
Э
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/b6A2S/A2S/current_q_network/current_q_network/out/b/Adam8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
use_nesterov( *
_output_shapes
:
ъ
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
_output_shapes
: *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
Ъ
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
ь
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
Ю
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( *
T0


A2S/Adam_2NoOpJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1


A2S/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/b6A2S/best_policy_network/best_policy_network/fc0/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
Ѕ
A2S/Assign_1Assign7A2S/current_policy_network/current_policy_network/fc0/w6A2S/best_policy_network/best_policy_network/fc0/w/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
Ё
A2S/Assign_2Assign7A2S/current_policy_network/current_policy_network/fc1/b6A2S/best_policy_network/best_policy_network/fc1/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
Ѕ
A2S/Assign_3Assign7A2S/current_policy_network/current_policy_network/fc1/w6A2S/best_policy_network/best_policy_network/fc1/w/read*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 
Ё
A2S/Assign_4Assign7A2S/current_policy_network/current_policy_network/out/b6A2S/best_policy_network/best_policy_network/out/b/read*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
Ѕ
A2S/Assign_5Assign7A2S/current_policy_network/current_policy_network/out/w6A2S/best_policy_network/best_policy_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(

A2S/Assign_6Assign5A2S/current_value_network/current_value_network/fc0/b4A2S/best_value_network/best_value_network/fc0/b/read*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0

A2S/Assign_7Assign5A2S/current_value_network/current_value_network/fc0/w4A2S/best_value_network/best_value_network/fc0/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w

A2S/Assign_8Assign5A2S/current_value_network/current_value_network/fc1/b4A2S/best_value_network/best_value_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b

A2S/Assign_9Assign5A2S/current_value_network/current_value_network/fc1/w4A2S/best_value_network/best_value_network/fc1/w/read*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@

A2S/Assign_10Assign5A2S/current_value_network/current_value_network/out/b4A2S/best_value_network/best_value_network/out/b/read*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
 
A2S/Assign_11Assign5A2S/current_value_network/current_value_network/out/w4A2S/best_value_network/best_value_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w

A2S/Assign_12Assign-A2S/current_q_network/current_q_network/fc0/b,A2S/best_q_network/best_q_network/fc0/b/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@

A2S/Assign_13Assign-A2S/current_q_network/current_q_network/fc0/w,A2S/best_q_network/best_q_network/fc0/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_14Assign-A2S/current_q_network/current_q_network/fc1/b,A2S/best_q_network/best_q_network/fc1/b/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@

A2S/Assign_15Assign-A2S/current_q_network/current_q_network/fc1/w,A2S/best_q_network/best_q_network/fc1/w/read*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0

A2S/Assign_16Assign-A2S/current_q_network/current_q_network/out/b,A2S/best_q_network/best_q_network/out/b/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:

A2S/Assign_17Assign-A2S/current_q_network/current_q_network/out/w,A2S/best_q_network/best_q_network/out/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
Њ
A2S/group_depsNoOp^A2S/Assign^A2S/Assign_1^A2S/Assign_2^A2S/Assign_3^A2S/Assign_4^A2S/Assign_5^A2S/Assign_6^A2S/Assign_7^A2S/Assign_8^A2S/Assign_9^A2S/Assign_10^A2S/Assign_11^A2S/Assign_12^A2S/Assign_13^A2S/Assign_14^A2S/Assign_15^A2S/Assign_16^A2S/Assign_17

A2S/Assign_18Assign1A2S/best_policy_network/best_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b
 
A2S/Assign_19Assign1A2S/best_policy_network/best_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_20Assign1A2S/best_policy_network/best_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b
 
A2S/Assign_21Assign1A2S/best_policy_network/best_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@

A2S/Assign_22Assign1A2S/best_policy_network/best_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
 
A2S/Assign_23Assign1A2S/best_policy_network/best_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_24Assign/A2S/best_value_network/best_value_network/fc0/b:A2S/current_value_network/current_value_network/fc0/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b

A2S/Assign_25Assign/A2S/best_value_network/best_value_network/fc0/w:A2S/current_value_network/current_value_network/fc0/w/read*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0

A2S/Assign_26Assign/A2S/best_value_network/best_value_network/fc1/b:A2S/current_value_network/current_value_network/fc1/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@

A2S/Assign_27Assign/A2S/best_value_network/best_value_network/fc1/w:A2S/current_value_network/current_value_network/fc1/w/read*
_output_shapes

:@@*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
validate_shape(

A2S/Assign_28Assign/A2S/best_value_network/best_value_network/out/b:A2S/current_value_network/current_value_network/out/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(*
_output_shapes
:

A2S/Assign_29Assign/A2S/best_value_network/best_value_network/out/w:A2S/current_value_network/current_value_network/out/w/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@
ў
A2S/Assign_30Assign'A2S/best_q_network/best_q_network/fc0/b2A2S/current_q_network/current_q_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(

A2S/Assign_31Assign'A2S/best_q_network/best_q_network/fc0/w2A2S/current_q_network/current_q_network/fc0/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
ў
A2S/Assign_32Assign'A2S/best_q_network/best_q_network/fc1/b2A2S/current_q_network/current_q_network/fc1/b/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@

A2S/Assign_33Assign'A2S/best_q_network/best_q_network/fc1/w2A2S/current_q_network/current_q_network/fc1/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ў
A2S/Assign_34Assign'A2S/best_q_network/best_q_network/out/b2A2S/current_q_network/current_q_network/out/b/read*
_output_shapes
:*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(

A2S/Assign_35Assign'A2S/best_q_network/best_q_network/out/w2A2S/current_q_network/current_q_network/out/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@
И
A2S/group_deps_1NoOp^A2S/Assign_18^A2S/Assign_19^A2S/Assign_20^A2S/Assign_21^A2S/Assign_22^A2S/Assign_23^A2S/Assign_24^A2S/Assign_25^A2S/Assign_26^A2S/Assign_27^A2S/Assign_28^A2S/Assign_29^A2S/Assign_30^A2S/Assign_31^A2S/Assign_32^A2S/Assign_33^A2S/Assign_34^A2S/Assign_35

A2S/Assign_36Assign1A2S/last_policy_network/last_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
 
A2S/Assign_37Assign1A2S/last_policy_network/last_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w

A2S/Assign_38Assign1A2S/last_policy_network/last_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
 
A2S/Assign_39Assign1A2S/last_policy_network/last_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@

A2S/Assign_40Assign1A2S/last_policy_network/last_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
 
A2S/Assign_41Assign1A2S/last_policy_network/last_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
validate_shape(
x
A2S/group_deps_2NoOp^A2S/Assign_36^A2S/Assign_37^A2S/Assign_38^A2S/Assign_39^A2S/Assign_40^A2S/Assign_41

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
A2S/Const_7Const*
dtype0*
_output_shapes
:*
valueB"       
m

A2S/Mean_4MeanA2S/advantagesA2S/Const_7*
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
A2S/Mean_4*
T0*
_output_shapes
: "PъыЭ\-     хљN	xНѓQXжAJЯк
нЛ
9
Add
x"T
y"T
z"T"
Ttype:
2	
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.3.02v1.3.0-rc2-20-g0787eeeп
s
A2S/observationsPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
n
A2S/actionsPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
A2S/advantagesPlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
V
A2S/learning_ratePlaceholder*
dtype0*
_output_shapes
:*
shape:
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
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
W
A2S/average_rewardPlaceholder*
_output_shapes
:*
shape:*
dtype0
ѕ
XA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB"   @   *
dtype0
ч
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
ч
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB
 *  ?
ц
`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/shape*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
seed2*
dtype0*
_output_shapes

:@
њ
VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes
: 

VA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
ў
RA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform/min*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@*
T0
ї
7A2S/current_policy_network/current_policy_network/fc0/w
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
ѓ
>A2S/current_policy_network/current_policy_network/fc0/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/wRA2S/current_policy_network/current_policy_network/fc0/w/Initializer/random_uniform*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
і
<A2S/current_policy_network/current_policy_network/fc0/w/readIdentity7A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w
т
IA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
я
7A2S/current_policy_network/current_policy_network/fc0/b
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
ц
>A2S/current_policy_network/current_policy_network/fc0/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/bIA2S/current_policy_network/current_policy_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(
ђ
<A2S/current_policy_network/current_policy_network/fc0/b/readIdentity7A2S/current_policy_network/current_policy_network/fc0/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
:@
г
!A2S/current_policy_network/MatMulMatMulA2S/observations<A2S/current_policy_network/current_policy_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
И
A2S/current_policy_network/addAdd!A2S/current_policy_network/MatMul<A2S/current_policy_network/current_policy_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
y
A2S/current_policy_network/TanhTanhA2S/current_policy_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
ѕ
XA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB"@   @   *
dtype0
ч
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  П
ч
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ц
`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
seed2
њ
VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes
: 

VA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
ў
RA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
ї
7A2S/current_policy_network/current_policy_network/fc1/w
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w
ѓ
>A2S/current_policy_network/current_policy_network/fc1/w/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/wRA2S/current_policy_network/current_policy_network/fc1/w/Initializer/random_uniform*
_output_shapes

:@@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(
і
<A2S/current_policy_network/current_policy_network/fc1/w/readIdentity7A2S/current_policy_network/current_policy_network/fc1/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
т
IA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
я
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
ц
>A2S/current_policy_network/current_policy_network/fc1/b/AssignAssign7A2S/current_policy_network/current_policy_network/fc1/bIA2S/current_policy_network/current_policy_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(
ђ
<A2S/current_policy_network/current_policy_network/fc1/b/readIdentity7A2S/current_policy_network/current_policy_network/fc1/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
ф
#A2S/current_policy_network/MatMul_1MatMulA2S/current_policy_network/Tanh<A2S/current_policy_network/current_policy_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
М
 A2S/current_policy_network/add_1Add#A2S/current_policy_network/MatMul_1<A2S/current_policy_network/current_policy_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
}
!A2S/current_policy_network/Tanh_1Tanh A2S/current_policy_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
ѕ
XA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shapeConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
ч
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/minConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
ч
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB
 *ЭЬЬ=
ц
`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformXA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
seed2-
њ
VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/subSubVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/maxVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes
: 

VA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulMul`A2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/RandomUniformVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ў
RA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniformAddVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/mulVA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w
ї
7A2S/current_policy_network/current_policy_network/out/w
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
ѓ
>A2S/current_policy_network/current_policy_network/out/w/AssignAssign7A2S/current_policy_network/current_policy_network/out/wRA2S/current_policy_network/current_policy_network/out/w/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@
і
<A2S/current_policy_network/current_policy_network/out/w/readIdentity7A2S/current_policy_network/current_policy_network/out/w*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
т
IA2S/current_policy_network/current_policy_network/out/b/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
я
7A2S/current_policy_network/current_policy_network/out/b
VariableV2*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:*
dtype0
ц
>A2S/current_policy_network/current_policy_network/out/b/AssignAssign7A2S/current_policy_network/current_policy_network/out/bIA2S/current_policy_network/current_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:
ђ
<A2S/current_policy_network/current_policy_network/out/b/readIdentity7A2S/current_policy_network/current_policy_network/out/b*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
_output_shapes
:
ц
#A2S/current_policy_network/MatMul_2MatMul!A2S/current_policy_network/Tanh_1<A2S/current_policy_network/current_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
М
 A2S/current_policy_network/add_2Add#A2S/current_policy_network/MatMul_2<A2S/current_policy_network/current_policy_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
щ
RA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
valueB"   @   *
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
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
seed2=
т
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w
є
PA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
ц
LA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
ы
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
л
8A2S/best_policy_network/best_policy_network/fc0/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/wLA2S/best_policy_network/best_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
ф
6A2S/best_policy_network/best_policy_network/fc0/w/readIdentity1A2S/best_policy_network/best_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
_output_shapes

:@
ж
CA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1A2S/best_policy_network/best_policy_network/fc0/b
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ю
8A2S/best_policy_network/best_policy_network/fc0/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc0/bCA2S/best_policy_network/best_policy_network/fc0/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
р
6A2S/best_policy_network/best_policy_network/fc0/b/readIdentity1A2S/best_policy_network/best_policy_network/fc0/b*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
_output_shapes
:@*
T0
Ъ
A2S/best_policy_network/MatMulMatMulA2S/observations6A2S/best_policy_network/best_policy_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Ќ
A2S/best_policy_network/addAddA2S/best_policy_network/MatMul6A2S/best_policy_network/best_policy_network/fc0/b/read*'
_output_shapes
:џџџџџџџџџ@*
T0
s
A2S/best_policy_network/TanhTanhA2S/best_policy_network/add*'
_output_shapes
:џџџџџџџџџ@*
T0
щ
RA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB"@   @   *
dtype0
л
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/minConst*
_output_shapes
: *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  П*
dtype0
л
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
д
ZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/shape*
seed2N*
dtype0*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w
т
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w
є
PA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
ц
LA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@*
T0
ы
1A2S/best_policy_network/best_policy_network/fc1/w
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
	container *
shape
:@@
л
8A2S/best_policy_network/best_policy_network/fc1/w/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/wLA2S/best_policy_network/best_policy_network/fc1/w/Initializer/random_uniform*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
ф
6A2S/best_policy_network/best_policy_network/fc1/w/readIdentity1A2S/best_policy_network/best_policy_network/fc1/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w*
_output_shapes

:@@
ж
CA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1A2S/best_policy_network/best_policy_network/fc1/b
VariableV2*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ю
8A2S/best_policy_network/best_policy_network/fc1/b/AssignAssign1A2S/best_policy_network/best_policy_network/fc1/bCA2S/best_policy_network/best_policy_network/fc1/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
р
6A2S/best_policy_network/best_policy_network/fc1/b/readIdentity1A2S/best_policy_network/best_policy_network/fc1/b*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
_output_shapes
:@*
T0
и
 A2S/best_policy_network/MatMul_1MatMulA2S/best_policy_network/Tanh6A2S/best_policy_network/best_policy_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
А
A2S/best_policy_network/add_1Add A2S/best_policy_network/MatMul_16A2S/best_policy_network/best_policy_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
w
A2S/best_policy_network/Tanh_1TanhA2S/best_policy_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
щ
RA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
valueB"@      *
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
dtype0*
_output_shapes

:@*

seed*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
seed2_
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

:@
ц
LA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniformAddPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/mulPA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
ы
1A2S/best_policy_network/best_policy_network/out/w
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
	container *
shape
:@
л
8A2S/best_policy_network/best_policy_network/out/w/AssignAssign1A2S/best_policy_network/best_policy_network/out/wLA2S/best_policy_network/best_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w
ф
6A2S/best_policy_network/best_policy_network/out/w/readIdentity1A2S/best_policy_network/best_policy_network/out/w*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
_output_shapes

:@
ж
CA2S/best_policy_network/best_policy_network/out/b/Initializer/zerosConst*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
у
1A2S/best_policy_network/best_policy_network/out/b
VariableV2*
shared_name *D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
Ю
8A2S/best_policy_network/best_policy_network/out/b/AssignAssign1A2S/best_policy_network/best_policy_network/out/bCA2S/best_policy_network/best_policy_network/out/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:
р
6A2S/best_policy_network/best_policy_network/out/b/readIdentity1A2S/best_policy_network/best_policy_network/out/b*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
_output_shapes
:
к
 A2S/best_policy_network/MatMul_2MatMulA2S/best_policy_network/Tanh_16A2S/best_policy_network/best_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
A2S/best_policy_network/add_2Add A2S/best_policy_network/MatMul_26A2S/best_policy_network/best_policy_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
щ
RA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
л
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
л
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
valueB
 *  ?
д
ZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
seed2o*
dtype0*
_output_shapes

:@*

seed
т
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes
: 
є
PA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
ц
LA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
ы
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
л
8A2S/last_policy_network/last_policy_network/fc0/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/wLA2S/last_policy_network/last_policy_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
ф
6A2S/last_policy_network/last_policy_network/fc0/w/readIdentity1A2S/last_policy_network/last_policy_network/fc0/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w*
_output_shapes

:@
ж
CA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zerosConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
у
1A2S/last_policy_network/last_policy_network/fc0/b
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b
Ю
8A2S/last_policy_network/last_policy_network/fc0/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc0/bCA2S/last_policy_network/last_policy_network/fc0/b/Initializer/zeros*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
р
6A2S/last_policy_network/last_policy_network/fc0/b/readIdentity1A2S/last_policy_network/last_policy_network/fc0/b*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
_output_shapes
:@
Ъ
A2S/last_policy_network/MatMulMatMulA2S/observations6A2S/last_policy_network/last_policy_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
Ќ
A2S/last_policy_network/addAddA2S/last_policy_network/MatMul6A2S/last_policy_network/last_policy_network/fc0/b/read*'
_output_shapes
:џџџџџџџџџ@*
T0
s
A2S/last_policy_network/TanhTanhA2S/last_policy_network/add*'
_output_shapes
:џџџџџџџџџ@*
T0
щ
RA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
л
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
л
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
е
ZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/shape*
_output_shapes

:@@*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
seed2*
dtype0
т
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w
є
PA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
ц
LA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
ы
1A2S/last_policy_network/last_policy_network/fc1/w
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w
л
8A2S/last_policy_network/last_policy_network/fc1/w/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/wLA2S/last_policy_network/last_policy_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ф
6A2S/last_policy_network/last_policy_network/fc1/w/readIdentity1A2S/last_policy_network/last_policy_network/fc1/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
_output_shapes

:@@
ж
CA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zerosConst*
_output_shapes
:@*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
valueB@*    *
dtype0
у
1A2S/last_policy_network/last_policy_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
	container *
shape:@
Ю
8A2S/last_policy_network/last_policy_network/fc1/b/AssignAssign1A2S/last_policy_network/last_policy_network/fc1/bCA2S/last_policy_network/last_policy_network/fc1/b/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
р
6A2S/last_policy_network/last_policy_network/fc1/b/readIdentity1A2S/last_policy_network/last_policy_network/fc1/b*
_output_shapes
:@*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b
и
 A2S/last_policy_network/MatMul_1MatMulA2S/last_policy_network/Tanh6A2S/last_policy_network/last_policy_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
А
A2S/last_policy_network/add_1Add A2S/last_policy_network/MatMul_16A2S/last_policy_network/last_policy_network/fc1/b/read*'
_output_shapes
:џџџџџџџџџ@*
T0
w
A2S/last_policy_network/Tanh_1TanhA2S/last_policy_network/add_1*'
_output_shapes
:џџџџџџџџџ@*
T0
щ
RA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shapeConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
л
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/minConst*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
л
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
valueB
 *ЭЬЬ=
е
ZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformRA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/shape*

seed*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
seed2*
dtype0*
_output_shapes

:@
т
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/subSubPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/maxPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes
: 
є
PA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulMulZA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/RandomUniformPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
ц
LA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniformAddPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/mulPA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform/min*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@*
T0
ы
1A2S/last_policy_network/last_policy_network/out/w
VariableV2*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@
л
8A2S/last_policy_network/last_policy_network/out/w/AssignAssign1A2S/last_policy_network/last_policy_network/out/wLA2S/last_policy_network/last_policy_network/out/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w
ф
6A2S/last_policy_network/last_policy_network/out/w/readIdentity1A2S/last_policy_network/last_policy_network/out/w*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
_output_shapes

:@
ж
CA2S/last_policy_network/last_policy_network/out/b/Initializer/zerosConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
valueB*    
у
1A2S/last_policy_network/last_policy_network/out/b
VariableV2*
shared_name *D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
Ю
8A2S/last_policy_network/last_policy_network/out/b/AssignAssign1A2S/last_policy_network/last_policy_network/out/bCA2S/last_policy_network/last_policy_network/out/b/Initializer/zeros*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
р
6A2S/last_policy_network/last_policy_network/out/b/readIdentity1A2S/last_policy_network/last_policy_network/out/b*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
_output_shapes
:*
T0
к
 A2S/last_policy_network/MatMul_2MatMulA2S/last_policy_network/Tanh_16A2S/last_policy_network/last_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
A2S/last_policy_network/add_2Add A2S/last_policy_network/MatMul_26A2S/last_policy_network/last_policy_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
ё
VA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
у
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
у
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
с
^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
seed2Ё
ђ
TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes
: 

TA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
і
PA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
ѓ
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
ы
<A2S/current_value_network/current_value_network/fc0/w/AssignAssign5A2S/current_value_network/current_value_network/fc0/wPA2S/current_value_network/current_value_network/fc0/w/Initializer/random_uniform*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
№
:A2S/current_value_network/current_value_network/fc0/w/readIdentity5A2S/current_value_network/current_value_network/fc0/w*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
о
GA2S/current_value_network/current_value_network/fc0/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ы
5A2S/current_value_network/current_value_network/fc0/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container 
о
<A2S/current_value_network/current_value_network/fc0/b/AssignAssign5A2S/current_value_network/current_value_network/fc0/bGA2S/current_value_network/current_value_network/fc0/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(
ь
:A2S/current_value_network/current_value_network/fc0/b/readIdentity5A2S/current_value_network/current_value_network/fc0/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
а
 A2S/current_value_network/MatMulMatMulA2S/observations:A2S/current_value_network/current_value_network/fc0/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
Д
A2S/current_value_network/addAdd A2S/current_value_network/MatMul:A2S/current_value_network/current_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
w
A2S/current_value_network/TanhTanhA2S/current_value_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
ё
VA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB"@   @   
у
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
у
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
с
^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
seed2В*
dtype0*
_output_shapes

:@@*

seed
ђ
TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes
: 

TA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
і
PA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
ѓ
5A2S/current_value_network/current_value_network/fc1/w
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
ы
<A2S/current_value_network/current_value_network/fc1/w/AssignAssign5A2S/current_value_network/current_value_network/fc1/wPA2S/current_value_network/current_value_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
№
:A2S/current_value_network/current_value_network/fc1/w/readIdentity5A2S/current_value_network/current_value_network/fc1/w*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
о
GA2S/current_value_network/current_value_network/fc1/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ы
5A2S/current_value_network/current_value_network/fc1/b
VariableV2*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
о
<A2S/current_value_network/current_value_network/fc1/b/AssignAssign5A2S/current_value_network/current_value_network/fc1/bGA2S/current_value_network/current_value_network/fc1/b/Initializer/zeros*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ь
:A2S/current_value_network/current_value_network/fc1/b/readIdentity5A2S/current_value_network/current_value_network/fc1/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
_output_shapes
:@
р
"A2S/current_value_network/MatMul_1MatMulA2S/current_value_network/Tanh:A2S/current_value_network/current_value_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
И
A2S/current_value_network/add_1Add"A2S/current_value_network/MatMul_1:A2S/current_value_network/current_value_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
{
 A2S/current_value_network/Tanh_1TanhA2S/current_value_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
ё
VA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
у
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/minConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
у
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
с
^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformVA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
seed2У
ђ
TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/subSubTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/maxTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes
: 

TA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulMul^A2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/RandomUniformTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
і
PA2S/current_value_network/current_value_network/out/w/Initializer/random_uniformAddTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/mulTA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
ѓ
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
ы
<A2S/current_value_network/current_value_network/out/w/AssignAssign5A2S/current_value_network/current_value_network/out/wPA2S/current_value_network/current_value_network/out/w/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(*
_output_shapes

:@
№
:A2S/current_value_network/current_value_network/out/w/readIdentity5A2S/current_value_network/current_value_network/out/w*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
о
GA2S/current_value_network/current_value_network/out/b/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ы
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
о
<A2S/current_value_network/current_value_network/out/b/AssignAssign5A2S/current_value_network/current_value_network/out/bGA2S/current_value_network/current_value_network/out/b/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(
ь
:A2S/current_value_network/current_value_network/out/b/readIdentity5A2S/current_value_network/current_value_network/out/b*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:
т
"A2S/current_value_network/MatMul_2MatMul A2S/current_value_network/Tanh_1:A2S/current_value_network/current_value_network/out/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
И
A2S/current_value_network/add_2Add"A2S/current_value_network/MatMul_2:A2S/current_value_network/current_value_network/out/b/read*'
_output_shapes
:џџџџџџџџџ*
T0
х
PA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shapeConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
valueB"   @   *
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
XA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
seed2г
к
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
ь
NA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@
о
JA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@*
T0
ч
/A2S/best_value_network/best_value_network/fc0/w
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
г
6A2S/best_value_network/best_value_network/fc0/w/AssignAssign/A2S/best_value_network/best_value_network/fc0/wJA2S/best_value_network/best_value_network/fc0/w/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w
о
4A2S/best_value_network/best_value_network/fc0/w/readIdentity/A2S/best_value_network/best_value_network/fc0/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
_output_shapes

:@
в
AA2S/best_value_network/best_value_network/fc0/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
п
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
Ц
6A2S/best_value_network/best_value_network/fc0/b/AssignAssign/A2S/best_value_network/best_value_network/fc0/bAA2S/best_value_network/best_value_network/fc0/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
validate_shape(*
_output_shapes
:@
к
4A2S/best_value_network/best_value_network/fc0/b/readIdentity/A2S/best_value_network/best_value_network/fc0/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b*
_output_shapes
:@
Ч
A2S/best_value_network/MatMulMatMulA2S/observations4A2S/best_value_network/best_value_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Ј
A2S/best_value_network/addAddA2S/best_value_network/MatMul4A2S/best_value_network/best_value_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
q
A2S/best_value_network/TanhTanhA2S/best_value_network/add*'
_output_shapes
:џџџџџџџџџ@*
T0
х
PA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB"@   @   
з
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/minConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
з
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Я
XA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/shape*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
seed2ф*
dtype0*
_output_shapes

:@@*

seed
к
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes
: 
ь
NA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
о
JA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@*
T0
ч
/A2S/best_value_network/best_value_network/fc1/w
VariableV2*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
г
6A2S/best_value_network/best_value_network/fc1/w/AssignAssign/A2S/best_value_network/best_value_network/fc1/wJA2S/best_value_network/best_value_network/fc1/w/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@
о
4A2S/best_value_network/best_value_network/fc1/w/readIdentity/A2S/best_value_network/best_value_network/fc1/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
_output_shapes

:@@
в
AA2S/best_value_network/best_value_network/fc1/b/Initializer/zerosConst*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
п
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
Ц
6A2S/best_value_network/best_value_network/fc1/b/AssignAssign/A2S/best_value_network/best_value_network/fc1/bAA2S/best_value_network/best_value_network/fc1/b/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
к
4A2S/best_value_network/best_value_network/fc1/b/readIdentity/A2S/best_value_network/best_value_network/fc1/b*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
_output_shapes
:@
д
A2S/best_value_network/MatMul_1MatMulA2S/best_value_network/Tanh4A2S/best_value_network/best_value_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
Ќ
A2S/best_value_network/add_1AddA2S/best_value_network/MatMul_14A2S/best_value_network/best_value_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
u
A2S/best_value_network/Tanh_1TanhA2S/best_value_network/add_1*'
_output_shapes
:џџџџџџџџџ@*
T0
х
PA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
valueB"@      *
dtype0
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
XA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformPA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/shape*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
seed2ѕ*
dtype0*
_output_shapes

:@*

seed*
T0
к
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/subSubNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/maxNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w
ь
NA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulMulXA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/RandomUniformNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/sub*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@*
T0
о
JA2S/best_value_network/best_value_network/out/w/Initializer/random_uniformAddNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/mulNA2S/best_value_network/best_value_network/out/w/Initializer/random_uniform/min*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@*
T0
ч
/A2S/best_value_network/best_value_network/out/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
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

:@
о
4A2S/best_value_network/best_value_network/out/w/readIdentity/A2S/best_value_network/best_value_network/out/w*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
_output_shapes

:@
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
VariableV2*
shared_name *B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
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
ж
A2S/best_value_network/MatMul_2MatMulA2S/best_value_network/Tanh_14A2S/best_value_network/best_value_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ќ
A2S/best_value_network/add_2AddA2S/best_value_network/MatMul_24A2S/best_value_network/best_value_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
h
A2S/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
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
Е
A2S/strided_sliceStridedSlice A2S/current_policy_network/add_2A2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
T0*
Index0*
shrink_axis_mask 
`
A2S/SqueezeSqueezeA2S/strided_slice*
squeeze_dims
 *
T0*
_output_shapes
:
b
A2S/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   
v
A2S/ReshapeReshapeA2S/SqueezeA2S/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
К
A2S/strided_slice_1StridedSliceA2S/best_policy_network/add_2A2S/strided_slice_1/stackA2S/strided_slice_1/stack_1A2S/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ
d
A2S/Squeeze_1SqueezeA2S/strided_slice_1*
T0*
_output_shapes
:*
squeeze_dims
 
d
A2S/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
|
A2S/Reshape_1ReshapeA2S/Squeeze_1A2S/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
j
A2S/strided_slice_2/stackConst*
_output_shapes
:*
valueB"        *
dtype0
l
A2S/strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
l
A2S/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
К
A2S/strided_slice_2StridedSliceA2S/last_policy_network/add_2A2S/strided_slice_2/stackA2S/strided_slice_2/stack_1A2S/strided_slice_2/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
T0*
Index0*
shrink_axis_mask 
d
A2S/Squeeze_2SqueezeA2S/strided_slice_2*
T0*
_output_shapes
:*
squeeze_dims
 
d
A2S/Reshape_2/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
|
A2S/Reshape_2ReshapeA2S/Squeeze_2A2S/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
N
	A2S/ConstConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
P
A2S/Const_1Const*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
P
A2S/Const_2Const*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
Y
A2S/Normal/locIdentityA2S/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
H
A2S/Normal/scaleIdentity	A2S/Const*
T0*
_output_shapes
: 
]
A2S/Normal_1/locIdentityA2S/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
L
A2S/Normal_1/scaleIdentityA2S/Const_1*
_output_shapes
: *
T0
]
A2S/Normal_2/locIdentityA2S/Reshape_2*'
_output_shapes
:џџџџџџџџџ*
T0
L
A2S/Normal_2/scaleIdentityA2S/Const_2*
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
,A2S/KullbackLeibler/kl_normal_normal/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   @
q
,A2S/KullbackLeibler/kl_normal_normal/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
h
+A2S/KullbackLeibler/kl_normal_normal/SquareSquareA2S/Normal/scale*
T0*
_output_shapes
: 
l
-A2S/KullbackLeibler/kl_normal_normal/Square_1SquareA2S/Normal_2/scale*
T0*
_output_shapes
: 
Д
,A2S/KullbackLeibler/kl_normal_normal/truedivRealDiv+A2S/KullbackLeibler/kl_normal_normal/Square-A2S/KullbackLeibler/kl_normal_normal/Square_1*
T0*
_output_shapes
: 

(A2S/KullbackLeibler/kl_normal_normal/subSubA2S/Normal/locA2S/Normal_2/loc*'
_output_shapes
:џџџџџџџџџ*
T0

-A2S/KullbackLeibler/kl_normal_normal/Square_2Square(A2S/KullbackLeibler/kl_normal_normal/sub*
T0*'
_output_shapes
:џџџџџџџџџ
­
(A2S/KullbackLeibler/kl_normal_normal/mulMul,A2S/KullbackLeibler/kl_normal_normal/Const_1-A2S/KullbackLeibler/kl_normal_normal/Square_1*
_output_shapes
: *
T0
Ф
.A2S/KullbackLeibler/kl_normal_normal/truediv_1RealDiv-A2S/KullbackLeibler/kl_normal_normal/Square_2(A2S/KullbackLeibler/kl_normal_normal/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
*A2S/KullbackLeibler/kl_normal_normal/sub_1Sub,A2S/KullbackLeibler/kl_normal_normal/truediv*A2S/KullbackLeibler/kl_normal_normal/Const*
T0*
_output_shapes
: 
~
(A2S/KullbackLeibler/kl_normal_normal/LogLog,A2S/KullbackLeibler/kl_normal_normal/truediv*
T0*
_output_shapes
: 
Ј
*A2S/KullbackLeibler/kl_normal_normal/sub_2Sub*A2S/KullbackLeibler/kl_normal_normal/sub_1(A2S/KullbackLeibler/kl_normal_normal/Log*
_output_shapes
: *
T0
Ќ
*A2S/KullbackLeibler/kl_normal_normal/mul_1Mul,A2S/KullbackLeibler/kl_normal_normal/Const_2*A2S/KullbackLeibler/kl_normal_normal/sub_2*
_output_shapes
: *
T0
Н
(A2S/KullbackLeibler/kl_normal_normal/addAdd.A2S/KullbackLeibler/kl_normal_normal/truediv_1*A2S/KullbackLeibler/kl_normal_normal/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0
\
A2S/Const_3Const*
valueB"       *
dtype0*
_output_shapes
:

A2S/MeanMean(A2S/KullbackLeibler/kl_normal_normal/addA2S/Const_3*
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
s
%A2S/Normal_3/batch_shape_tensor/ShapeShapeA2S/Normal/loc*
_output_shapes
:*
T0*
out_type0
j
'A2S/Normal_3/batch_shape_tensor/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Г
-A2S/Normal_3/batch_shape_tensor/BroadcastArgsBroadcastArgs%A2S/Normal_3/batch_shape_tensor/Shape'A2S/Normal_3/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
]
A2S/concat/values_0Const*
_output_shapes
:*
valueB:*
dtype0
Q
A2S/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ѕ

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
 :џџџџџџџџџџџџџџџџџџ*
seed2Л*

seed

A2S/random_normal/mulMul&A2S/random_normal/RandomStandardNormalA2S/random_normal/stddev*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0

A2S/random_normalAddA2S/random_normal/mulA2S/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
r
A2S/mulMulA2S/random_normalA2S/Normal/scale*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
]
A2S/addAddA2S/mulA2S/Normal/loc*
T0*+
_output_shapes
:џџџџџџџџџ
h
A2S/Reshape_3/shapeConst*!
valueB"џџџџ      *
dtype0*
_output_shapes
:
z
A2S/Reshape_3ReshapeA2S/addA2S/Reshape_3/shape*
T0*
Tshape0*+
_output_shapes
:џџџџџџџџџ
S
A2S/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B :

A2S/concat_1ConcatV2A2S/observationsA2S/actionsA2S/concat_1/axis*'
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0*
N
с
NA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB"   @   *
dtype0*
_output_shapes
:
г
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
г
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Щ
VA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
seed2Ч
в
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes
: 
ф
LA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
ж
HA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
у
-A2S/current_q_network/current_q_network/fc0/w
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
Ы
4A2S/current_q_network/current_q_network/fc0/w/AssignAssign-A2S/current_q_network/current_q_network/fc0/wHA2S/current_q_network/current_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
и
2A2S/current_q_network/current_q_network/fc0/w/readIdentity-A2S/current_q_network/current_q_network/fc0/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
Ю
?A2S/current_q_network/current_q_network/fc0/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
л
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
О
4A2S/current_q_network/current_q_network/fc0/b/AssignAssign-A2S/current_q_network/current_q_network/fc0/b?A2S/current_q_network/current_q_network/fc0/b/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
д
2A2S/current_q_network/current_q_network/fc0/b/readIdentity-A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
Р
A2S/current_q_network/MatMulMatMulA2S/concat_12A2S/current_q_network/current_q_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
Є
A2S/current_q_network/addAddA2S/current_q_network/MatMul2A2S/current_q_network/current_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
o
A2S/current_q_network/TanhTanhA2S/current_q_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
с
NA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB"@   @   *
dtype0*
_output_shapes
:
г
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  П
г
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB
 *  ?
Щ
VA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/shape*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
seed2и*
dtype0*
_output_shapes

:@@*

seed*
T0
в
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes
: 
ф
LA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
ж
HA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
у
-A2S/current_q_network/current_q_network/fc1/w
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
Ы
4A2S/current_q_network/current_q_network/fc1/w/AssignAssign-A2S/current_q_network/current_q_network/fc1/wHA2S/current_q_network/current_q_network/fc1/w/Initializer/random_uniform*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(
и
2A2S/current_q_network/current_q_network/fc1/w/readIdentity-A2S/current_q_network/current_q_network/fc1/w*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
Ю
?A2S/current_q_network/current_q_network/fc1/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
л
-A2S/current_q_network/current_q_network/fc1/b
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
	container *
shape:@
О
4A2S/current_q_network/current_q_network/fc1/b/AssignAssign-A2S/current_q_network/current_q_network/fc1/b?A2S/current_q_network/current_q_network/fc1/b/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(
д
2A2S/current_q_network/current_q_network/fc1/b/readIdentity-A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
а
A2S/current_q_network/MatMul_1MatMulA2S/current_q_network/Tanh2A2S/current_q_network/current_q_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
Ј
A2S/current_q_network/add_1AddA2S/current_q_network/MatMul_12A2S/current_q_network/current_q_network/fc1/b/read*'
_output_shapes
:џџџџџџџџџ@*
T0
s
A2S/current_q_network/Tanh_1TanhA2S/current_q_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
с
NA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shapeConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB"@      *
dtype0*
_output_shapes
:
г
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/minConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *ЭЬЬН*
dtype0*
_output_shapes
: 
г
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Щ
VA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformNA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
seed2щ*
dtype0*
_output_shapes

:@*

seed
в
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/subSubLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/maxLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes
: 
ф
LA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulMulVA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/RandomUniformLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
ж
HA2S/current_q_network/current_q_network/out/w/Initializer/random_uniformAddLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/mulLA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform/min*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
у
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
Ы
4A2S/current_q_network/current_q_network/out/w/AssignAssign-A2S/current_q_network/current_q_network/out/wHA2S/current_q_network/current_q_network/out/w/Initializer/random_uniform*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
и
2A2S/current_q_network/current_q_network/out/w/readIdentity-A2S/current_q_network/current_q_network/out/w*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@*
T0
Ю
?A2S/current_q_network/current_q_network/out/b/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
л
-A2S/current_q_network/current_q_network/out/b
VariableV2*
_output_shapes
:*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:*
dtype0
О
4A2S/current_q_network/current_q_network/out/b/AssignAssign-A2S/current_q_network/current_q_network/out/b?A2S/current_q_network/current_q_network/out/b/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(
д
2A2S/current_q_network/current_q_network/out/b/readIdentity-A2S/current_q_network/current_q_network/out/b*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:
в
A2S/current_q_network/MatMul_2MatMulA2S/current_q_network/Tanh_12A2S/current_q_network/current_q_network/out/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ј
A2S/current_q_network/add_2AddA2S/current_q_network/MatMul_22A2S/current_q_network/current_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
е
HA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
valueB"   @   *
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

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
seed2љ*
dtype0*
_output_shapes

:@
К
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
Ь
FA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
О
BA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform/min*
_output_shapes

:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w
з
'A2S/best_q_network/best_q_network/fc0/w
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
	container 
Г
.A2S/best_q_network/best_q_network/fc0/w/AssignAssign'A2S/best_q_network/best_q_network/fc0/wBA2S/best_q_network/best_q_network/fc0/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
Ц
,A2S/best_q_network/best_q_network/fc0/w/readIdentity'A2S/best_q_network/best_q_network/fc0/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
_output_shapes

:@
Т
9A2S/best_q_network/best_q_network/fc0/b/Initializer/zerosConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
Я
'A2S/best_q_network/best_q_network/fc0/b
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
І
.A2S/best_q_network/best_q_network/fc0/b/AssignAssign'A2S/best_q_network/best_q_network/fc0/b9A2S/best_q_network/best_q_network/fc0/b/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b*
validate_shape(*
_output_shapes
:@
Т
,A2S/best_q_network/best_q_network/fc0/b/readIdentity'A2S/best_q_network/best_q_network/fc0/b*
_output_shapes
:@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b
З
A2S/best_q_network/MatMulMatMulA2S/concat_1,A2S/best_q_network/best_q_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0

A2S/best_q_network/addAddA2S/best_q_network/MatMul,A2S/best_q_network/best_q_network/fc0/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
i
A2S/best_q_network/TanhTanhA2S/best_q_network/add*
T0*'
_output_shapes
:џџџџџџџџџ@
е
HA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB"@   @   
Ч
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/minConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  П*
dtype0*
_output_shapes
: 
Ч
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
valueB
 *  ?*
dtype0*
_output_shapes
: 
З
PA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
seed2
К
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/subSubFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/maxFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes
: 
Ь
FA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulMulPA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/RandomUniformFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w
О
BA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
з
'A2S/best_q_network/best_q_network/fc1/w
VariableV2*
_output_shapes

:@@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
	container *
shape
:@@*
dtype0
Г
.A2S/best_q_network/best_q_network/fc1/w/AssignAssign'A2S/best_q_network/best_q_network/fc1/wBA2S/best_q_network/best_q_network/fc1/w/Initializer/random_uniform*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
Ц
,A2S/best_q_network/best_q_network/fc1/w/readIdentity'A2S/best_q_network/best_q_network/fc1/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
_output_shapes

:@@
Т
9A2S/best_q_network/best_q_network/fc1/b/Initializer/zerosConst*
_output_shapes
:@*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
valueB@*    *
dtype0
Я
'A2S/best_q_network/best_q_network/fc1/b
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
	container 
І
.A2S/best_q_network/best_q_network/fc1/b/AssignAssign'A2S/best_q_network/best_q_network/fc1/b9A2S/best_q_network/best_q_network/fc1/b/Initializer/zeros*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(
Т
,A2S/best_q_network/best_q_network/fc1/b/readIdentity'A2S/best_q_network/best_q_network/fc1/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
_output_shapes
:@
Ф
A2S/best_q_network/MatMul_1MatMulA2S/best_q_network/Tanh,A2S/best_q_network/best_q_network/fc1/w/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 

A2S/best_q_network/add_1AddA2S/best_q_network/MatMul_1,A2S/best_q_network/best_q_network/fc1/b/read*
T0*'
_output_shapes
:џџџџџџџџџ@
m
A2S/best_q_network/Tanh_1TanhA2S/best_q_network/add_1*
T0*'
_output_shapes
:џџџџџџџџџ@
е
HA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB"@      *
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
FA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/maxConst*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
З
PA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/RandomUniformRandomUniformHA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes

:@*

seed*
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

:@
О
BA2S/best_q_network/best_q_network/out/w/Initializer/random_uniformAddFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/mulFA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@
з
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
Г
.A2S/best_q_network/best_q_network/out/w/AssignAssign'A2S/best_q_network/best_q_network/out/wBA2S/best_q_network/best_q_network/out/w/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@
Ц
,A2S/best_q_network/best_q_network/out/w/readIdentity'A2S/best_q_network/best_q_network/out/w*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
_output_shapes

:@
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
,A2S/best_q_network/best_q_network/out/b/readIdentity'A2S/best_q_network/best_q_network/out/b*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
_output_shapes
:
Ц
A2S/best_q_network/MatMul_2MatMulA2S/best_q_network/Tanh_1,A2S/best_q_network/best_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

A2S/best_q_network/add_2AddA2S/best_q_network/MatMul_2,A2S/best_q_network/best_q_network/out/b/read*
T0*'
_output_shapes
:џџџџџџџџџ
{
%A2S/Normal_4/log_prob/standardize/subSubA2S/actionsA2S/Normal/loc*
T0*'
_output_shapes
:џџџџџџџџџ

)A2S/Normal_4/log_prob/standardize/truedivRealDiv%A2S/Normal_4/log_prob/standardize/subA2S/Normal/scale*'
_output_shapes
:џџџџџџџџџ*
T0

A2S/Normal_4/log_prob/SquareSquare)A2S/Normal_4/log_prob/standardize/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
`
A2S/Normal_4/log_prob/mul/xConst*
valueB
 *   П*
dtype0*
_output_shapes
: 

A2S/Normal_4/log_prob/mulMulA2S/Normal_4/log_prob/mul/xA2S/Normal_4/log_prob/Square*'
_output_shapes
:џџџџџџџџџ*
T0
S
A2S/Normal_4/log_prob/LogLogA2S/Normal/scale*
T0*
_output_shapes
: 
`
A2S/Normal_4/log_prob/add/xConst*
_output_shapes
: *
valueB
 *?k?*
dtype0
y
A2S/Normal_4/log_prob/addAddA2S/Normal_4/log_prob/add/xA2S/Normal_4/log_prob/Log*
_output_shapes
: *
T0

A2S/Normal_4/log_prob/subSubA2S/Normal_4/log_prob/mulA2S/Normal_4/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
[
A2S/NegNegA2S/Normal_4/log_prob/sub*
T0*'
_output_shapes
:џџџџџџџџџ
[
	A2S/mul_1MulA2S/NegA2S/advantages*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_4Const*
dtype0*
_output_shapes
:*
valueB"       
h

A2S/Mean_1Mean	A2S/mul_1A2S/Const_4*
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
A2S/Mean_1*
T0*
_output_shapes
: 

A2S/SquaredDifferenceSquaredDifferenceA2S/current_value_network/add_2A2S/returns*
T0*'
_output_shapes
:џџџџџџџџџ
\
A2S/Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
t

A2S/Mean_2MeanA2S/SquaredDifferenceA2S/Const_5*
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
A2S/Mean_2*
T0*
_output_shapes
: 

A2S/SquaredDifference_1SquaredDifferenceA2S/current_q_network/add_2A2S/returns*
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

A2S/Mean_3MeanA2S/SquaredDifference_1A2S/Const_6*
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
 *  ?*
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
valueB"      *
dtype0*
_output_shapes
:
Ј
%A2S/gradients/A2S/Mean_1_grad/ReshapeReshapeA2S/gradients/Fill+A2S/gradients/A2S/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
l
#A2S/gradients/A2S/Mean_1_grad/ShapeShape	A2S/mul_1*
_output_shapes
:*
T0*
out_type0
К
"A2S/gradients/A2S/Mean_1_grad/TileTile%A2S/gradients/A2S/Mean_1_grad/Reshape#A2S/gradients/A2S/Mean_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
n
%A2S/gradients/A2S/Mean_1_grad/Shape_1Shape	A2S/mul_1*
_output_shapes
:*
T0*
out_type0
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
Д
"A2S/gradients/A2S/Mean_1_grad/ProdProd%A2S/gradients/A2S/Mean_1_grad/Shape_1#A2S/gradients/A2S/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
o
%A2S/gradients/A2S/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
И
$A2S/gradients/A2S/Mean_1_grad/Prod_1Prod%A2S/gradients/A2S/Mean_1_grad/Shape_2%A2S/gradients/A2S/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
'A2S/gradients/A2S/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
%A2S/gradients/A2S/Mean_1_grad/MaximumMaximum$A2S/gradients/A2S/Mean_1_grad/Prod_1'A2S/gradients/A2S/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

&A2S/gradients/A2S/Mean_1_grad/floordivFloorDiv"A2S/gradients/A2S/Mean_1_grad/Prod%A2S/gradients/A2S/Mean_1_grad/Maximum*
_output_shapes
: *
T0

"A2S/gradients/A2S/Mean_1_grad/CastCast&A2S/gradients/A2S/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Њ
%A2S/gradients/A2S/Mean_1_grad/truedivRealDiv"A2S/gradients/A2S/Mean_1_grad/Tile"A2S/gradients/A2S/Mean_1_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
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
в
2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"A2S/gradients/A2S/mul_1_grad/Shape$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

 A2S/gradients/A2S/mul_1_grad/mulMul%A2S/gradients/A2S/Mean_1_grad/truedivA2S/advantages*'
_output_shapes
:џџџџџџџџџ*
T0
Н
 A2S/gradients/A2S/mul_1_grad/SumSum A2S/gradients/A2S/mul_1_grad/mul2A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Е
$A2S/gradients/A2S/mul_1_grad/ReshapeReshape A2S/gradients/A2S/mul_1_grad/Sum"A2S/gradients/A2S/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

"A2S/gradients/A2S/mul_1_grad/mul_1MulA2S/Neg%A2S/gradients/A2S/Mean_1_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
У
"A2S/gradients/A2S/mul_1_grad/Sum_1Sum"A2S/gradients/A2S/mul_1_grad/mul_14A2S/gradients/A2S/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Л
&A2S/gradients/A2S/mul_1_grad/Reshape_1Reshape"A2S/gradients/A2S/mul_1_grad/Sum_1$A2S/gradients/A2S/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

-A2S/gradients/A2S/mul_1_grad/tuple/group_depsNoOp%^A2S/gradients/A2S/mul_1_grad/Reshape'^A2S/gradients/A2S/mul_1_grad/Reshape_1

5A2S/gradients/A2S/mul_1_grad/tuple/control_dependencyIdentity$A2S/gradients/A2S/mul_1_grad/Reshape.^A2S/gradients/A2S/mul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ

2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ShapeShapeA2S/Normal_4/log_prob/mul*
T0*
out_type0*
_output_shapes
:
w
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

BA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
л
0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/SumSumA2S/gradients/A2S/Neg_grad/NegBA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
х
4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
п
2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1SumA2S/gradients/A2S/Neg_grad/NegDA2S/gradients/A2S/Normal_4/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/NegNeg2A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
и
6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1Reshape0A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Neg4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
=A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1
Т
EA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
З
GA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/sub_grad/Reshape_1
u
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 

4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1ShapeA2S/Normal_4/log_prob/Square*
out_type0*
_output_shapes
:*
T0

BA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ю
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulMulEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependencyA2S/Normal_4/log_prob/Square*'
_output_shapes
:џџџџџџџџџ*
T0
э
0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/SumSum0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mulBA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
д
4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/ReshapeReshape0A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Я
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1MulA2S/Normal_4/log_prob/mul/xEA2S/gradients/A2S/Normal_4/log_prob/sub_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
ѓ
2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_1Sum2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/mul_1DA2S/gradients/A2S/Normal_4/log_prob/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1Reshape2A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Sum_14A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Е
=A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_depsNoOp5^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape7^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1
Б
EA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependencyIdentity4A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*G
_class=
;9loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape
Ш
GA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1Identity6A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1>^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients/A2S/Normal_4/log_prob/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ф
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/xConstH^A2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *   @
Ю
3A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul/x)A2S/Normal_4/log_prob/standardize/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
ь
5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1MulGA2S/gradients/A2S/Normal_4/log_prob/mul_grad/tuple/control_dependency_13A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeShape%A2S/Normal_4/log_prob/standardize/sub*
_output_shapes
:*
T0*
out_type0

DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
В
RA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ShapeDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRealDiv5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1A2S/Normal/scale*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumSumDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDivRA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeReshape@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/SumBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
 
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegNeg%A2S/Normal_4/log_prob/standardize/sub*
T0*'
_output_shapes
:џџџџџџџџџ
з
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1RealDiv@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/NegA2S/Normal/scale*
T0*'
_output_shapes
:џџџџџџџџџ
н
FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2RealDivFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_1A2S/Normal/scale*'
_output_shapes
:џџџџџџџџџ*
T0
ј
@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulMul5A2S/gradients/A2S/Normal_4/log_prob/Square_grad/mul_1FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/RealDiv_2*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
BA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1Sum@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/mulTA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

FA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1ReshapeBA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Sum_1DA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
х
MA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_depsNoOpE^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeG^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1

UA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentityDA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/ReshapeN^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*W
_classM
KIloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ї
WA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependency_1IdentityFA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1N^A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/Reshape_1*
_output_shapes
: 

>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ShapeShapeA2S/actions*
_output_shapes
:*
T0*
out_type0

@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1ShapeA2S/Normal/loc*
_output_shapes
:*
T0*
out_type0
І
NA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Њ
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/SumSumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyNA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeReshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ў
>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1SumUA2S/gradients/A2S/Normal_4/log_prob/standardize/truediv_grad/tuple/control_dependencyPA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
І
<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/NegNeg>A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:

BA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1Reshape<A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Neg@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
й
IA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_depsNoOpA^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeC^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1
ђ
QA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/ReshapeJ^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape
ј
SA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityBA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1J^A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/group_deps*U
_classK
IGloc:@A2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
x
$A2S/gradients/A2S/Reshape_grad/ShapeShapeA2S/Squeeze*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
н
&A2S/gradients/A2S/Reshape_grad/ReshapeReshapeSA2S/gradients/A2S/Normal_4/log_prob/standardize/sub_grad/tuple/control_dependency_1$A2S/gradients/A2S/Reshape_grad/Shape*
_output_shapes
:*
T0*
Tshape0
u
$A2S/gradients/A2S/Squeeze_grad/ShapeShapeA2S/strided_slice*
_output_shapes
:*
T0*
out_type0
П
&A2S/gradients/A2S/Squeeze_grad/ReshapeReshape&A2S/gradients/A2S/Reshape_grad/Reshape$A2S/gradients/A2S/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

*A2S/gradients/A2S/strided_slice_grad/ShapeShape A2S/current_policy_network/add_2*
_output_shapes
:*
T0*
out_type0

5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradStridedSliceGrad*A2S/gradients/A2S/strided_slice_grad/ShapeA2S/strided_slice/stackA2S/strided_slice/stack_1A2S/strided_slice/stack_2&A2S/gradients/A2S/Squeeze_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask

9A2S/gradients/A2S/current_policy_network/add_2_grad/ShapeShape#A2S/current_policy_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

IA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

7A2S/gradients/A2S/current_policy_network/add_2_grad/SumSum5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradIA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
њ
;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_2_grad/Sum9A2S/gradients/A2S/current_policy_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1Sum5A2S/gradients/A2S/strided_slice_grad/StridedSliceGradKA2S/gradients/A2S/current_policy_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ѓ
=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_2_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
Ъ
DA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1
о
LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_2_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
з
NA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_2_grad/Reshape_1*
_output_shapes
:
Ћ
=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(

?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1MatMul!A2S/current_policy_network/Tanh_1LA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
б
GA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1
ш
OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul
х
QA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/group_deps*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@*
T0
я
=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradTanhGrad!A2S/current_policy_network/Tanh_1OA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ@*
T0

9A2S/gradients/A2S/current_policy_network/add_1_grad/ShapeShape#A2S/current_policy_network/MatMul_1*
_output_shapes
:*
T0*
out_type0

;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:

IA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

7A2S/gradients/A2S/current_policy_network/add_1_grad/SumSum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
њ
;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeReshape7A2S/gradients/A2S/current_policy_network/add_1_grad/Sum9A2S/gradients/A2S/current_policy_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1Sum=A2S/gradients/A2S/current_policy_network/Tanh_1_grad/TanhGradKA2S/gradients/A2S/current_policy_network/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ѓ
=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1Reshape9A2S/gradients/A2S/current_policy_network/add_1_grad/Sum_1;A2S/gradients/A2S/current_policy_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Ъ
DA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape>^A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1
о
LA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/add_1_grad/ReshapeE^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
з
NA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1E^A2S/gradients/A2S/current_policy_network/add_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/add_1_grad/Reshape_1
Ћ
=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulMatMulLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(

?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1MatMulA2S/current_policy_network/TanhLA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
б
GA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_depsNoOp>^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul@^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1
ш
OA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependencyIdentity=A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMulH^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul
х
QA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1Identity?A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1H^A2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/group_deps*
_output_shapes

:@@*
T0*R
_classH
FDloc:@A2S/gradients/A2S/current_policy_network/MatMul_1_grad/MatMul_1
ы
;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradTanhGradA2S/current_policy_network/TanhOA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ@*
T0

7A2S/gradients/A2S/current_policy_network/add_grad/ShapeShape!A2S/current_policy_network/MatMul*
_output_shapes
:*
T0*
out_type0

9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:

GA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs7A2S/gradients/A2S/current_policy_network/add_grad/Shape9A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5A2S/gradients/A2S/current_policy_network/add_grad/SumSum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradGA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
є
9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeReshape5A2S/gradients/A2S/current_policy_network/add_grad/Sum7A2S/gradients/A2S/current_policy_network/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0

7A2S/gradients/A2S/current_policy_network/add_grad/Sum_1Sum;A2S/gradients/A2S/current_policy_network/Tanh_grad/TanhGradIA2S/gradients/A2S/current_policy_network/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
э
;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1Reshape7A2S/gradients/A2S/current_policy_network/add_grad/Sum_19A2S/gradients/A2S/current_policy_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Ф
BA2S/gradients/A2S/current_policy_network/add_grad/tuple/group_depsNoOp:^A2S/gradients/A2S/current_policy_network/add_grad/Reshape<^A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1
ж
JA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependencyIdentity9A2S/gradients/A2S/current_policy_network/add_grad/ReshapeC^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*L
_classB
@>loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape
Я
LA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1Identity;A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1C^A2S/gradients/A2S/current_policy_network/add_grad/tuple/group_deps*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/add_grad/Reshape_1*
_output_shapes
:@*
T0
Ї
;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulMatMulJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency<A2S/current_policy_network/current_policy_network/fc0/w/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
є
=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1MatMulA2S/observationsJA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
Ы
EA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_depsNoOp<^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul>^A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1
р
MA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependencyIdentity;A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMulF^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
н
OA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1Identity=A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1F^A2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@A2S/gradients/A2S/current_policy_network/MatMul_grad/MatMul_1*
_output_shapes

:@
Ў
A2S/beta1_power/initial_valueConst*
valueB
 *fff?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
П
A2S/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape: 
ц
A2S/beta1_power/AssignAssignA2S/beta1_powerA2S/beta1_power/initial_value*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 

A2S/beta1_power/readIdentityA2S/beta1_power*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: *
T0
Ў
A2S/beta2_power/initial_valueConst*
valueB
 *wО?*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
dtype0*
_output_shapes
: 
П
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
ц
A2S/beta2_power/AssignAssignA2S/beta2_powerA2S/beta2_power/initial_value*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: 

A2S/beta2_power/readIdentityA2S/beta2_power*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: 
ѓ
RA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@

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

GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@

EA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
ѕ
TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@

BA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1
VariableV2*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
	container *
shape
:@*
dtype0

IA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@

GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
_output_shapes

:@
ы
RA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ј
@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
	container *
shape:@

GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@

EA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam*
_output_shapes
:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
э
TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
њ
BA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b

IA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@

GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1*
_output_shapes
:@*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b
ѓ
RA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zerosConst*
_output_shapes

:@@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0

@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam
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

GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w

EA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@
ѕ
TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
valueB@@*    *
dtype0

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

IA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/Initializer/zeros*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0

GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
_output_shapes

:@@*
T0
ы
RA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ј
@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container *
shape:@

GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamRA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/Initializer/zeros*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

EA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
э
TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zerosConst*
_output_shapes
:@*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
valueB@*    *
dtype0
њ
BA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
	container 

IA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@

GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
_output_shapes
:@
ѓ
RA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@

@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam
VariableV2*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@

GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamRA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@

EA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/w/Adam*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ѕ
TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@

BA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
	container *
shape
:@

IA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
validate_shape(*
_output_shapes

:@

GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
_output_shapes

:@
ы
RA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ј
@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
	container *
shape:

GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/AssignAssign@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamRA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:

EA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/readIdentity@A2S/A2S/current_policy_network/current_policy_network/out/b/Adam*
_output_shapes
:*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
э
TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zerosConst*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
њ
BA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b

IA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/AssignAssignBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1TA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
validate_shape(*
_output_shapes
:

GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/readIdentityBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1*
_output_shapes
:*
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
ћ
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/w@A2S/A2S/current_policy_network/current_policy_network/fc0/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonOA2S/gradients/A2S/current_policy_network/MatMul_grad/tuple/control_dependency_1*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0
є
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc0/b@A2S/A2S/current_policy_network/current_policy_network/fc0/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonLA2S/gradients/A2S/current_policy_network/add_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
use_nesterov( 
§
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/w@A2S/A2S/current_policy_network/current_policy_network/fc1/w/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
use_nesterov( *
_output_shapes

:@@
і
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/fc1/b@A2S/A2S/current_policy_network/current_policy_network/fc1/b/AdamBA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
use_nesterov( *
_output_shapes
:@
§
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/w@A2S/A2S/current_policy_network/current_policy_network/out/w/AdamBA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonQA2S/gradients/A2S/current_policy_network/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w*
use_nesterov( *
_output_shapes

:@
і
QA2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam	ApplyAdam7A2S/current_policy_network/current_policy_network/out/b@A2S/A2S/current_policy_network/current_policy_network/out/b/AdamBA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1A2S/beta1_power/readA2S/beta2_power/readA2S/learning_rateA2S/Adam/beta1A2S/Adam/beta2A2S/Adam/epsilonNA2S/gradients/A2S/current_policy_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b*
use_nesterov( *
_output_shapes
:

A2S/Adam/mulMulA2S/beta1_power/readA2S/Adam/beta1R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: *
T0
Ю
A2S/Adam/AssignAssignA2S/beta1_powerA2S/Adam/mul*
_output_shapes
: *
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(
 
A2S/Adam/mul_1MulA2S/beta2_power/readA2S/Adam/beta2R^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
_output_shapes
: *
T0
в
A2S/Adam/Assign_1AssignA2S/beta2_powerA2S/Adam/mul_1*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
: *
use_locking( 
Ў
A2S/AdamNoOpR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc0/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/fc1/b/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/w/ApplyAdamR^A2S/Adam/update_A2S/current_policy_network/current_policy_network/out/b/ApplyAdam^A2S/Adam/Assign^A2S/Adam/Assign_1
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
Ў
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
Р
$A2S/gradients_1/A2S/Mean_2_grad/TileTile'A2S/gradients_1/A2S/Mean_2_grad/Reshape%A2S/gradients_1/A2S/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
%A2S/gradients_1/A2S/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
К
$A2S/gradients_1/A2S/Mean_2_grad/ProdProd'A2S/gradients_1/A2S/Mean_2_grad/Shape_1%A2S/gradients_1/A2S/Mean_2_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
q
'A2S/gradients_1/A2S/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
О
&A2S/gradients_1/A2S/Mean_2_grad/Prod_1Prod'A2S/gradients_1/A2S/Mean_2_grad/Shape_2'A2S/gradients_1/A2S/Mean_2_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
k
)A2S/gradients_1/A2S/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'A2S/gradients_1/A2S/Mean_2_grad/MaximumMaximum&A2S/gradients_1/A2S/Mean_2_grad/Prod_1)A2S/gradients_1/A2S/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
Є
(A2S/gradients_1/A2S/Mean_2_grad/floordivFloorDiv$A2S/gradients_1/A2S/Mean_2_grad/Prod'A2S/gradients_1/A2S/Mean_2_grad/Maximum*
_output_shapes
: *
T0

$A2S/gradients_1/A2S/Mean_2_grad/CastCast(A2S/gradients_1/A2S/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_1/A2S/Mean_2_grad/truedivRealDiv$A2S/gradients_1/A2S/Mean_2_grad/Tile$A2S/gradients_1/A2S/Mean_2_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0

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
ќ
@A2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs0A2S/gradients_1/A2S/SquaredDifference_grad/Shape2A2S/gradients_1/A2S/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
 
1A2S/gradients_1/A2S/SquaredDifference_grad/scalarConst(^A2S/gradients_1/A2S/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
У
.A2S/gradients_1/A2S/SquaredDifference_grad/mulMul1A2S/gradients_1/A2S/SquaredDifference_grad/scalar'A2S/gradients_1/A2S/Mean_2_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
П
.A2S/gradients_1/A2S/SquaredDifference_grad/subSubA2S/current_value_network/add_2A2S/returns(^A2S/gradients_1/A2S/Mean_2_grad/truediv*'
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
:*

Tidx0*
	keep_dims( *
T0
п
2A2S/gradients_1/A2S/SquaredDifference_grad/ReshapeReshape.A2S/gradients_1/A2S/SquaredDifference_grad/Sum0A2S/gradients_1/A2S/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
э
0A2S/gradients_1/A2S/SquaredDifference_grad/Sum_1Sum0A2S/gradients_1/A2S/SquaredDifference_grad/mul_1BA2S/gradients_1/A2S/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
EA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependency_1Identity.A2S/gradients_1/A2S/SquaredDifference_grad/Neg<^A2S/gradients_1/A2S/SquaredDifference_grad/tuple/group_deps*
T0*A
_class7
53loc:@A2S/gradients_1/A2S/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ

:A2S/gradients_1/A2S/current_value_network/add_2_grad/ShapeShape"A2S/current_value_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

JA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

8A2S/gradients_1/A2S/current_value_network/add_2_grad/SumSumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyJA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
§
<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1SumCA2S/gradients_1/A2S/SquaredDifference_grad/tuple/control_dependencyLA2S/gradients_1/A2S/current_value_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
і
>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_2_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Э
EA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1
т
MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_2_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape
л
OA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_2_grad/Reshape_1*
_output_shapes
:
Ћ
>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/out/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(*
T0

@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1MatMul A2S/current_value_network/Tanh_1MA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
д
HA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1
ь
PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
щ
RA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@
№
>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradTanhGrad A2S/current_value_network/Tanh_1PA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ@*
T0

:A2S/gradients_1/A2S/current_value_network/add_1_grad/ShapeShape"A2S/current_value_network/MatMul_1*
out_type0*
_output_shapes
:*
T0

<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:

JA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

8A2S/gradients_1/A2S/current_value_network/add_1_grad/SumSum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
§
<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeReshape8A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum:A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1Sum>A2S/gradients_1/A2S/current_value_network/Tanh_1_grad/TanhGradLA2S/gradients_1/A2S/current_value_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
і
>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1Reshape:A2S/gradients_1/A2S/current_value_network/add_1_grad/Sum_1<A2S/gradients_1/A2S/current_value_network/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Э
EA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape?^A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1
т
MA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/add_1_grad/ReshapeF^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape
л
OA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1F^A2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/add_1_grad/Reshape_1*
_output_shapes
:@
Ћ
>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulMatMulMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc1/w/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(*
T0

@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1MatMulA2S/current_value_network/TanhMA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@@*
transpose_a(
д
HA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_depsNoOp?^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulA^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1
ь
PA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependencyIdentity>A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMulI^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
щ
RA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1Identity@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1I^A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@A2S/gradients_1/A2S/current_value_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@
ь
<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradTanhGradA2S/current_value_network/TanhPA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

8A2S/gradients_1/A2S/current_value_network/add_grad/ShapeShape A2S/current_value_network/MatMul*
T0*
out_type0*
_output_shapes
:

:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@

HA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs8A2S/gradients_1/A2S/current_value_network/add_grad/Shape:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

6A2S/gradients_1/A2S/current_value_network/add_grad/SumSum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradHA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ї
:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeReshape6A2S/gradients_1/A2S/current_value_network/add_grad/Sum8A2S/gradients_1/A2S/current_value_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@

8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1Sum<A2S/gradients_1/A2S/current_value_network/Tanh_grad/TanhGradJA2S/gradients_1/A2S/current_value_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
№
<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1Reshape8A2S/gradients_1/A2S/current_value_network/add_grad/Sum_1:A2S/gradients_1/A2S/current_value_network/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
Ч
CA2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_depsNoOp;^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape=^A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1
к
KA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependencyIdentity:A2S/gradients_1/A2S/current_value_network/add_grad/ReshapeD^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*M
_classC
A?loc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@*
T0
г
MA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1Identity<A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1D^A2S/gradients_1/A2S/current_value_network/add_grad/tuple/group_deps*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/add_grad/Reshape_1*
_output_shapes
:@*
T0
Ї
<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulMatMulKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency:A2S/current_value_network/current_value_network/fc0/w/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
і
>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1MatMulA2S/observationsKA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
Ю
FA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_depsNoOp=^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul?^A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1
ф
NA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependencyIdentity<A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMulG^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
с
PA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1Identity>A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1G^A2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@A2S/gradients_1/A2S/current_value_network/MatMul_grad/MatMul_1*
_output_shapes

:@
Ў
A2S/beta1_power_1/initial_valueConst*
valueB
 *fff?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0*
_output_shapes
: 
П
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
ъ
A2S/beta1_power_1/AssignAssignA2S/beta1_power_1A2S/beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
 
A2S/beta1_power_1/readIdentityA2S/beta1_power_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
Ў
A2S/beta2_power_1/initial_valueConst*
valueB
 *wО?*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
dtype0*
_output_shapes
: 
П
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
ъ
A2S/beta2_power_1/AssignAssignA2S/beta2_power_1A2S/beta2_power_1/initial_value*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
 
A2S/beta2_power_1/readIdentityA2S/beta2_power_1*
_output_shapes
: *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
я
PA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
ќ
>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam
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
§
EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/w/AdamPA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(*
_output_shapes

:@

CA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@*
T0
ё
RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
valueB@*    
ў
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

GA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(

EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
_output_shapes

:@
ч
PA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    
є
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
љ
EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc0/b/AdamPA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(
ў
CA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
щ
RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
і
@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1
VariableV2*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
џ
GA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
:@

EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
:@
я
PA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
ќ
>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam
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
§
EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/w/AdamPA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking(

CA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam*
_output_shapes

:@@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w
ё
RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
valueB@@*    *
dtype0
ў
@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
	container 

GA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@

EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
_output_shapes

:@@
ч
PA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
є
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
љ
EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/fc1/b/AdamPA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@
ў
CA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
щ
RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
і
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
џ
GA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1RA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@

EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1*
_output_shapes
:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b
я
PA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ќ
>A2S/A2S/current_value_network/current_value_network/out/w/Adam
VariableV2*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
§
EA2S/A2S/current_value_network/current_value_network/out/w/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/w/AdamPA2S/A2S/current_value_network/current_value_network/out/w/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
validate_shape(

CA2S/A2S/current_value_network/current_value_network/out/w/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/w/Adam*
_output_shapes

:@*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w
ё
RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ў
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

GA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1RA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w

EA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
_output_shapes

:@
ч
PA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    
є
>A2S/A2S/current_value_network/current_value_network/out/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container *
shape:
љ
EA2S/A2S/current_value_network/current_value_network/out/b/Adam/AssignAssign>A2S/A2S/current_value_network/current_value_network/out/b/AdamPA2S/A2S/current_value_network/current_value_network/out/b/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(*
_output_shapes
:
ў
CA2S/A2S/current_value_network/current_value_network/out/b/Adam/readIdentity>A2S/A2S/current_value_network/current_value_network/out/b/Adam*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
_output_shapes
:*
T0
щ
RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
і
@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1
VariableV2*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
џ
GA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/AssignAssign@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1RA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b*
validate_shape(

EA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/readIdentity@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1*
_output_shapes
:*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b
U
A2S/Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
A2S/Adam_1/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wО?
W
A2S/Adam_1/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ў
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/w>A2S/A2S/current_value_network/current_value_network/fc0/w/Adam@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonPA2S/gradients_1/A2S/current_value_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
use_nesterov( *
_output_shapes

:@
ї
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc0/b>A2S/A2S/current_value_network/current_value_network/fc0/b/Adam@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonMA2S/gradients_1/A2S/current_value_network/add_grad/tuple/control_dependency_1*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0

QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/w>A2S/A2S/current_value_network/current_value_network/fc1/w/Adam@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_1_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
љ
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/fc1/b>A2S/A2S/current_value_network/current_value_network/fc1/b/Adam@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
use_nesterov( *
_output_shapes
:@

QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/w>A2S/A2S/current_value_network/current_value_network/out/w/Adam@A2S/A2S/current_value_network/current_value_network/out/w/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonRA2S/gradients_1/A2S/current_value_network/MatMul_2_grad/tuple/control_dependency_1*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w*
use_nesterov( 
љ
QA2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam	ApplyAdam5A2S/current_value_network/current_value_network/out/b>A2S/A2S/current_value_network/current_value_network/out/b/Adam@A2S/A2S/current_value_network/current_value_network/out/b/Adam_1A2S/beta1_power_1/readA2S/beta2_power_1/readA2S/learning_rateA2S/Adam_1/beta1A2S/Adam_1/beta2A2S/Adam_1/epsilonOA2S/gradients_1/A2S/current_value_network/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b
Ђ
A2S/Adam_1/mulMulA2S/beta1_power_1/readA2S/Adam_1/beta1R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
в
A2S/Adam_1/AssignAssignA2S/beta1_power_1A2S/Adam_1/mul*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(*
_output_shapes
: 
Є
A2S/Adam_1/mul_1MulA2S/beta2_power_1/readA2S/Adam_1/beta2R^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
_output_shapes
: 
ж
A2S/Adam_1/Assign_1AssignA2S/beta2_power_1A2S/Adam_1/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b
Д

A2S/Adam_1NoOpR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc0/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/fc1/b/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/w/ApplyAdamR^A2S/Adam_1/update_A2S/current_value_network/current_value_network/out/b/ApplyAdam^A2S/Adam_1/Assign^A2S/Adam_1/Assign_1
X
A2S/gradients_2/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
A2S/gradients_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
Ў
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
Р
$A2S/gradients_2/A2S/Mean_3_grad/TileTile'A2S/gradients_2/A2S/Mean_3_grad/Reshape%A2S/gradients_2/A2S/Mean_3_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
~
'A2S/gradients_2/A2S/Mean_3_grad/Shape_1ShapeA2S/SquaredDifference_1*
_output_shapes
:*
T0*
out_type0
j
'A2S/gradients_2/A2S/Mean_3_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%A2S/gradients_2/A2S/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
К
$A2S/gradients_2/A2S/Mean_3_grad/ProdProd'A2S/gradients_2/A2S/Mean_3_grad/Shape_1%A2S/gradients_2/A2S/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
'A2S/gradients_2/A2S/Mean_3_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
О
&A2S/gradients_2/A2S/Mean_3_grad/Prod_1Prod'A2S/gradients_2/A2S/Mean_3_grad/Shape_2'A2S/gradients_2/A2S/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
k
)A2S/gradients_2/A2S/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'A2S/gradients_2/A2S/Mean_3_grad/MaximumMaximum&A2S/gradients_2/A2S/Mean_3_grad/Prod_1)A2S/gradients_2/A2S/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
Є
(A2S/gradients_2/A2S/Mean_3_grad/floordivFloorDiv$A2S/gradients_2/A2S/Mean_3_grad/Prod'A2S/gradients_2/A2S/Mean_3_grad/Maximum*
_output_shapes
: *
T0

$A2S/gradients_2/A2S/Mean_3_grad/CastCast(A2S/gradients_2/A2S/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
А
'A2S/gradients_2/A2S/Mean_3_grad/truedivRealDiv$A2S/gradients_2/A2S/Mean_3_grad/Tile$A2S/gradients_2/A2S/Mean_3_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0

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

BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape4A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ђ
3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalarConst(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
Ч
0A2S/gradients_2/A2S/SquaredDifference_1_grad/mulMul3A2S/gradients_2/A2S/SquaredDifference_1_grad/scalar'A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Н
0A2S/gradients_2/A2S/SquaredDifference_1_grad/subSubA2S/current_q_network/add_2A2S/returns(^A2S/gradients_2/A2S/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Я
2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1Mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/mul0A2S/gradients_2/A2S/SquaredDifference_1_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
я
0A2S/gradients_2/A2S/SquaredDifference_1_grad/SumSum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1BA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
х
4A2S/gradients_2/A2S/SquaredDifference_1_grad/ReshapeReshape0A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ѓ
2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_1Sum2A2S/gradients_2/A2S/SquaredDifference_1_grad/mul_1DA2S/gradients_2/A2S/SquaredDifference_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1Reshape2A2S/gradients_2/A2S/SquaredDifference_1_grad/Sum_14A2S/gradients_2/A2S/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ё
0A2S/gradients_2/A2S/SquaredDifference_1_grad/NegNeg6A2S/gradients_2/A2S/SquaredDifference_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
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

6A2S/gradients_2/A2S/current_q_network/add_2_grad/ShapeShapeA2S/current_q_network/MatMul_2*
T0*
out_type0*
_output_shapes
:

8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

FA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

4A2S/gradients_2/A2S/current_q_network/add_2_grad/SumSumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyFA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ё
8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_1SumEA2S/gradients_2/A2S/SquaredDifference_1_grad/tuple/control_dependencyHA2S/gradients_2/A2S/current_q_network/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ъ
:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_2_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_2_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
С
AA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1
в
IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_2_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ы
KA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_2_grad/Reshape_1*
_output_shapes
:

:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/out/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
ў
<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1MatMulA2S/current_q_network/Tanh_1IA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
Ш
DA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1
м
LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
й
NA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/group_deps*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_2_grad/MatMul_1*
_output_shapes

:@*
T0
ф
:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradTanhGradA2S/current_q_network/Tanh_1LA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

6A2S/gradients_2/A2S/current_q_network/add_1_grad/ShapeShapeA2S/current_q_network/MatMul_1*
T0*
out_type0*
_output_shapes
:

8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:

FA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape8A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
џ
4A2S/gradients_2/A2S/current_q_network/add_1_grad/SumSum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ё
8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeReshape4A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum6A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0

6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_1Sum:A2S/gradients_2/A2S/current_q_network/Tanh_1_grad/TanhGradHA2S/gradients_2/A2S/current_q_network/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ъ
:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1Reshape6A2S/gradients_2/A2S/current_q_network/add_1_grad/Sum_18A2S/gradients_2/A2S/current_q_network/add_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
С
AA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape;^A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1
в
IA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/add_1_grad/ReshapeB^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
Ы
KA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1B^A2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/add_1_grad/Reshape_1*
_output_shapes
:@

:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulMatMulIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc1/w/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
ќ
<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1MatMulA2S/current_q_network/TanhIA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
Ш
DA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_depsNoOp;^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul=^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1
м
LA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependencyIdentity:A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMulE^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@
й
NA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1Identity<A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1E^A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/group_deps*O
_classE
CAloc:@A2S/gradients_2/A2S/current_q_network/MatMul_1_grad/MatMul_1*
_output_shapes

:@@*
T0
р
8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradTanhGradA2S/current_q_network/TanhLA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ@

4A2S/gradients_2/A2S/current_q_network/add_grad/ShapeShapeA2S/current_q_network/MatMul*
out_type0*
_output_shapes
:*
T0

6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:

DA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgsBroadcastGradientArgs4A2S/gradients_2/A2S/current_q_network/add_grad/Shape6A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
2A2S/gradients_2/A2S/current_q_network/add_grad/SumSum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradDA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
6A2S/gradients_2/A2S/current_q_network/add_grad/ReshapeReshape2A2S/gradients_2/A2S/current_q_network/add_grad/Sum4A2S/gradients_2/A2S/current_q_network/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ@
§
4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_1Sum8A2S/gradients_2/A2S/current_q_network/Tanh_grad/TanhGradFA2S/gradients_2/A2S/current_q_network/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ф
8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1Reshape4A2S/gradients_2/A2S/current_q_network/add_grad/Sum_16A2S/gradients_2/A2S/current_q_network/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
Л
?A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_depsNoOp7^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape9^A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1
Ъ
GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependencyIdentity6A2S/gradients_2/A2S/current_q_network/add_grad/Reshape@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ@
У
IA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1Identity8A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1@^A2S/gradients_2/A2S/current_q_network/add_grad/tuple/group_deps*
_output_shapes
:@*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/add_grad/Reshape_1

8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulMatMulGA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency2A2S/current_q_network/current_q_network/fc0/w/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ъ
:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1MatMulA2S/concat_1GA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
Т
BA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_depsNoOp9^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul;^A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1
д
JA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependencyIdentity8A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMulC^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
б
LA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1Identity:A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1C^A2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@A2S/gradients_2/A2S/current_q_network/MatMul_grad/MatMul_1*
_output_shapes

:@
І
A2S/beta1_power_2/initial_valueConst*
valueB
 *fff?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
З
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
т
A2S/beta1_power_2/AssignAssignA2S/beta1_power_2A2S/beta1_power_2/initial_value*
_output_shapes
: *
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(

A2S/beta1_power_2/readIdentityA2S/beta1_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
І
A2S/beta2_power_2/initial_valueConst*
valueB
 *wО?*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
dtype0*
_output_shapes
: 
З
A2S/beta2_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
	container *
shape: 
т
A2S/beta2_power_2/AssignAssignA2S/beta2_power_2A2S/beta2_power_2/initial_value*
_output_shapes
: *
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(

A2S/beta2_power_2/readIdentityA2S/beta2_power_2*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
п
HA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
ь
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
н
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/w/AdamHA2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@
ъ
;A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
с
JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
valueB@*    *
dtype0*
_output_shapes

:@
ю
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
у
?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking(
ю
=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
_output_shapes

:@
з
HA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    *
dtype0*
_output_shapes
:@
ф
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
й
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc0/b/AdamHA2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b
ц
;A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@
й
JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
valueB@*    
ц
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
п
?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking(
ъ
=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
:@
п
HA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0*
_output_shapes

:@@
ь
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
н
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/w/AdamHA2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Initializer/zeros*
_output_shapes

:@@*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(
ъ
;A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
с
JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zerosConst*
_output_shapes

:@@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
valueB@@*    *
dtype0
ю
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
у
?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ю
=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
_output_shapes

:@@
з
HA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    
ф
6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b
й
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/fc1/b/AdamHA2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
ц
;A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@
й
JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
valueB@*    *
dtype0*
_output_shapes
:@
ц
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
п
?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1JA2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@
ъ
=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
_output_shapes
:@
п
HA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:@*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    
ь
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
н
=A2S/A2S/current_q_network/current_q_network/out/w/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/w/AdamHA2S/A2S/current_q_network/current_q_network/out/w/Adam/Initializer/zeros*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(
ъ
;A2S/A2S/current_q_network/current_q_network/out/w/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/w/Adam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
_output_shapes

:@
с
JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
valueB@*    *
dtype0*
_output_shapes

:@
ю
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
у
?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1JA2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
ю
=A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1*
_output_shapes

:@*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w
з
HA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ф
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
й
=A2S/A2S/current_q_network/current_q_network/out/b/Adam/AssignAssign6A2S/A2S/current_q_network/current_q_network/out/b/AdamHA2S/A2S/current_q_network/current_q_network/out/b/Adam/Initializer/zeros*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ц
;A2S/A2S/current_q_network/current_q_network/out/b/Adam/readIdentity6A2S/A2S/current_q_network/current_q_network/out/b/Adam*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
_output_shapes
:*
T0
й
JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zerosConst*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
valueB*    *
dtype0*
_output_shapes
:
ц
8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
	container *
shape:*
dtype0*
_output_shapes
:
п
?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/AssignAssign8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1JA2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
validate_shape(*
_output_shapes
:
ъ
=A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/readIdentity8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1*
_output_shapes
:*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b
U
A2S/Adam_2/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
U
A2S/Adam_2/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wО?
W
A2S/Adam_2/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
в
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/w6A2S/A2S/current_q_network/current_q_network/fc0/w/Adam8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonLA2S/gradients_2/A2S/current_q_network/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
use_nesterov( *
_output_shapes

:@
Ы
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc0/b6A2S/A2S/current_q_network/current_q_network/fc0/b/Adam8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonIA2S/gradients_2/A2S/current_q_network/add_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
use_nesterov( *
_output_shapes
:@
д
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/w6A2S/A2S/current_q_network/current_q_network/fc1/w/Adam8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_1_grad/tuple/control_dependency_1*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
Э
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/fc1/b6A2S/A2S/current_q_network/current_q_network/fc1/b/Adam8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
use_nesterov( *
_output_shapes
:@
д
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/w6A2S/A2S/current_q_network/current_q_network/out/w/Adam8A2S/A2S/current_q_network/current_q_network/out/w/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonNA2S/gradients_2/A2S/current_q_network/MatMul_2_grad/tuple/control_dependency_1*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0
Э
IA2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam	ApplyAdam-A2S/current_q_network/current_q_network/out/b6A2S/A2S/current_q_network/current_q_network/out/b/Adam8A2S/A2S/current_q_network/current_q_network/out/b/Adam_1A2S/beta1_power_2/readA2S/beta2_power_2/readA2S/learning_rateA2S/Adam_2/beta1A2S/Adam_2/beta2A2S/Adam_2/epsilonKA2S/gradients_2/A2S/current_q_network/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b*
use_nesterov( *
_output_shapes
:
ъ
A2S/Adam_2/mulMulA2S/beta1_power_2/readA2S/Adam_2/beta1J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
Ъ
A2S/Adam_2/AssignAssignA2S/beta1_power_2A2S/Adam_2/mul*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(*
_output_shapes
: 
ь
A2S/Adam_2/mul_1MulA2S/beta2_power_2/readA2S/Adam_2/beta2J^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
_output_shapes
: 
Ю
A2S/Adam_2/Assign_1AssignA2S/beta2_power_2A2S/Adam_2/mul_1*
_output_shapes
: *
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b*
validate_shape(


A2S/Adam_2NoOpJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc0/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/fc1/b/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/w/ApplyAdamJ^A2S/Adam_2/update_A2S/current_q_network/current_q_network/out/b/ApplyAdam^A2S/Adam_2/Assign^A2S/Adam_2/Assign_1


A2S/AssignAssign7A2S/current_policy_network/current_policy_network/fc0/b6A2S/best_policy_network/best_policy_network/fc0/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@
Ѕ
A2S/Assign_1Assign7A2S/current_policy_network/current_policy_network/fc0/w6A2S/best_policy_network/best_policy_network/fc0/w/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@
Ё
A2S/Assign_2Assign7A2S/current_policy_network/current_policy_network/fc1/b6A2S/best_policy_network/best_policy_network/fc1/b/read*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
Ѕ
A2S/Assign_3Assign7A2S/current_policy_network/current_policy_network/fc1/w6A2S/best_policy_network/best_policy_network/fc1/w/read*
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 
Ё
A2S/Assign_4Assign7A2S/current_policy_network/current_policy_network/out/b6A2S/best_policy_network/best_policy_network/out/b/read*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/b
Ѕ
A2S/Assign_5Assign7A2S/current_policy_network/current_policy_network/out/w6A2S/best_policy_network/best_policy_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*J
_class@
><loc:@A2S/current_policy_network/current_policy_network/out/w

A2S/Assign_6Assign5A2S/current_value_network/current_value_network/fc0/b4A2S/best_value_network/best_value_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/b*
validate_shape(

A2S/Assign_7Assign5A2S/current_value_network/current_value_network/fc0/w4A2S/best_value_network/best_value_network/fc0/w/read*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc0/w*
validate_shape(

A2S/Assign_8Assign5A2S/current_value_network/current_value_network/fc1/b4A2S/best_value_network/best_value_network/fc1/b/read*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0

A2S/Assign_9Assign5A2S/current_value_network/current_value_network/fc1/w4A2S/best_value_network/best_value_network/fc1/w/read*
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 

A2S/Assign_10Assign5A2S/current_value_network/current_value_network/out/b4A2S/best_value_network/best_value_network/out/b/read*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/b
 
A2S/Assign_11Assign5A2S/current_value_network/current_value_network/out/w4A2S/best_value_network/best_value_network/out/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*H
_class>
<:loc:@A2S/current_value_network/current_value_network/out/w

A2S/Assign_12Assign-A2S/current_q_network/current_q_network/fc0/b,A2S/best_q_network/best_q_network/fc0/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/b

A2S/Assign_13Assign-A2S/current_q_network/current_q_network/fc0/w,A2S/best_q_network/best_q_network/fc0/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc0/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_14Assign-A2S/current_q_network/current_q_network/fc1/b,A2S/best_q_network/best_q_network/fc1/b/read*
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/b*
validate_shape(*
_output_shapes
:@*
use_locking( 

A2S/Assign_15Assign-A2S/current_q_network/current_q_network/fc1/w,A2S/best_q_network/best_q_network/fc1/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@

A2S/Assign_16Assign-A2S/current_q_network/current_q_network/out/b,A2S/best_q_network/best_q_network/out/b/read*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/b

A2S/Assign_17Assign-A2S/current_q_network/current_q_network/out/w,A2S/best_q_network/best_q_network/out/w/read*
use_locking( *
T0*@
_class6
42loc:@A2S/current_q_network/current_q_network/out/w*
validate_shape(*
_output_shapes

:@
Њ
A2S/group_depsNoOp^A2S/Assign^A2S/Assign_1^A2S/Assign_2^A2S/Assign_3^A2S/Assign_4^A2S/Assign_5^A2S/Assign_6^A2S/Assign_7^A2S/Assign_8^A2S/Assign_9^A2S/Assign_10^A2S/Assign_11^A2S/Assign_12^A2S/Assign_13^A2S/Assign_14^A2S/Assign_15^A2S/Assign_16^A2S/Assign_17

A2S/Assign_18Assign1A2S/best_policy_network/best_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/b*
validate_shape(*
_output_shapes
:@*
use_locking( 
 
A2S/Assign_19Assign1A2S/best_policy_network/best_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc0/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_20Assign1A2S/best_policy_network/best_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
 
A2S/Assign_21Assign1A2S/best_policy_network/best_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/fc1/w

A2S/Assign_22Assign1A2S/best_policy_network/best_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
 
A2S/Assign_23Assign1A2S/best_policy_network/best_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/best_policy_network/best_policy_network/out/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_24Assign/A2S/best_value_network/best_value_network/fc0/b:A2S/current_value_network/current_value_network/fc0/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/b

A2S/Assign_25Assign/A2S/best_value_network/best_value_network/fc0/w:A2S/current_value_network/current_value_network/fc0/w/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc0/w*
validate_shape(*
_output_shapes

:@

A2S/Assign_26Assign/A2S/best_value_network/best_value_network/fc1/b:A2S/current_value_network/current_value_network/fc1/b/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/b*
validate_shape(*
_output_shapes
:@

A2S/Assign_27Assign/A2S/best_value_network/best_value_network/fc1/w:A2S/current_value_network/current_value_network/fc1/w/read*
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( 

A2S/Assign_28Assign/A2S/best_value_network/best_value_network/out/b:A2S/current_value_network/current_value_network/out/b/read*
_output_shapes
:*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/b*
validate_shape(

A2S/Assign_29Assign/A2S/best_value_network/best_value_network/out/w:A2S/current_value_network/current_value_network/out/w/read*
use_locking( *
T0*B
_class8
64loc:@A2S/best_value_network/best_value_network/out/w*
validate_shape(*
_output_shapes

:@
ў
A2S/Assign_30Assign'A2S/best_q_network/best_q_network/fc0/b2A2S/current_q_network/current_q_network/fc0/b/read*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/b

A2S/Assign_31Assign'A2S/best_q_network/best_q_network/fc0/w2A2S/current_q_network/current_q_network/fc0/w/read*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc0/w*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
ў
A2S/Assign_32Assign'A2S/best_q_network/best_q_network/fc1/b2A2S/current_q_network/current_q_network/fc1/b/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/b*
validate_shape(*
_output_shapes
:@

A2S/Assign_33Assign'A2S/best_q_network/best_q_network/fc1/w2A2S/current_q_network/current_q_network/fc1/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/fc1/w*
validate_shape(*
_output_shapes

:@@
ў
A2S/Assign_34Assign'A2S/best_q_network/best_q_network/out/b2A2S/current_q_network/current_q_network/out/b/read*
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 

A2S/Assign_35Assign'A2S/best_q_network/best_q_network/out/w2A2S/current_q_network/current_q_network/out/w/read*
use_locking( *
T0*:
_class0
.,loc:@A2S/best_q_network/best_q_network/out/w*
validate_shape(*
_output_shapes

:@
И
A2S/group_deps_1NoOp^A2S/Assign_18^A2S/Assign_19^A2S/Assign_20^A2S/Assign_21^A2S/Assign_22^A2S/Assign_23^A2S/Assign_24^A2S/Assign_25^A2S/Assign_26^A2S/Assign_27^A2S/Assign_28^A2S/Assign_29^A2S/Assign_30^A2S/Assign_31^A2S/Assign_32^A2S/Assign_33^A2S/Assign_34^A2S/Assign_35

A2S/Assign_36Assign1A2S/last_policy_network/last_policy_network/fc0/b<A2S/current_policy_network/current_policy_network/fc0/b/read*
_output_shapes
:@*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/b*
validate_shape(
 
A2S/Assign_37Assign1A2S/last_policy_network/last_policy_network/fc0/w<A2S/current_policy_network/current_policy_network/fc0/w/read*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc0/w

A2S/Assign_38Assign1A2S/last_policy_network/last_policy_network/fc1/b<A2S/current_policy_network/current_policy_network/fc1/b/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/b*
validate_shape(*
_output_shapes
:@
 
A2S/Assign_39Assign1A2S/last_policy_network/last_policy_network/fc1/w<A2S/current_policy_network/current_policy_network/fc1/w/read*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/fc1/w*
validate_shape(*
_output_shapes

:@@*
use_locking( *
T0

A2S/Assign_40Assign1A2S/last_policy_network/last_policy_network/out/b<A2S/current_policy_network/current_policy_network/out/b/read*
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/b*
validate_shape(*
_output_shapes
:*
use_locking( 
 
A2S/Assign_41Assign1A2S/last_policy_network/last_policy_network/out/w<A2S/current_policy_network/current_policy_network/out/w/read*
use_locking( *
T0*D
_class:
86loc:@A2S/last_policy_network/last_policy_network/out/w*
validate_shape(*
_output_shapes

:@
x
A2S/group_deps_2NoOp^A2S/Assign_36^A2S/Assign_37^A2S/Assign_38^A2S/Assign_39^A2S/Assign_40^A2S/Assign_41

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
A2S/Const_7Const*
valueB"       *
dtype0*
_output_shapes
:
m

A2S/Mean_4MeanA2S/advantagesA2S/Const_7*
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
A2S/Mean_4*
T0*
_output_shapes
: ""
	summaries

A2S/kl:0
A2S/policy_network_loss:0
A2S/value_network_loss:0
A2S/q_network_loss:0
A2S/average_reward_1:0
A2S/average_advantage:0"ч7
trainable_variablesЯ7Ь7
Л
9A2S/current_policy_network/current_policy_network/fc0/w:0>A2S/current_policy_network/current_policy_network/fc0/w/Assign>A2S/current_policy_network/current_policy_network/fc0/w/read:0
Л
9A2S/current_policy_network/current_policy_network/fc0/b:0>A2S/current_policy_network/current_policy_network/fc0/b/Assign>A2S/current_policy_network/current_policy_network/fc0/b/read:0
Л
9A2S/current_policy_network/current_policy_network/fc1/w:0>A2S/current_policy_network/current_policy_network/fc1/w/Assign>A2S/current_policy_network/current_policy_network/fc1/w/read:0
Л
9A2S/current_policy_network/current_policy_network/fc1/b:0>A2S/current_policy_network/current_policy_network/fc1/b/Assign>A2S/current_policy_network/current_policy_network/fc1/b/read:0
Л
9A2S/current_policy_network/current_policy_network/out/w:0>A2S/current_policy_network/current_policy_network/out/w/Assign>A2S/current_policy_network/current_policy_network/out/w/read:0
Л
9A2S/current_policy_network/current_policy_network/out/b:0>A2S/current_policy_network/current_policy_network/out/b/Assign>A2S/current_policy_network/current_policy_network/out/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc1/w:08A2S/best_policy_network/best_policy_network/fc1/w/Assign8A2S/best_policy_network/best_policy_network/fc1/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc1/b:08A2S/best_policy_network/best_policy_network/fc1/b/Assign8A2S/best_policy_network/best_policy_network/fc1/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc0/w:08A2S/last_policy_network/last_policy_network/fc0/w/Assign8A2S/last_policy_network/last_policy_network/fc0/w/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc0/b:08A2S/last_policy_network/last_policy_network/fc0/b/Assign8A2S/last_policy_network/last_policy_network/fc0/b/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc1/w:08A2S/last_policy_network/last_policy_network/fc1/w/Assign8A2S/last_policy_network/last_policy_network/fc1/w/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc1/b:08A2S/last_policy_network/last_policy_network/fc1/b/Assign8A2S/last_policy_network/last_policy_network/fc1/b/read:0
Љ
3A2S/last_policy_network/last_policy_network/out/w:08A2S/last_policy_network/last_policy_network/out/w/Assign8A2S/last_policy_network/last_policy_network/out/w/read:0
Љ
3A2S/last_policy_network/last_policy_network/out/b:08A2S/last_policy_network/last_policy_network/out/b/Assign8A2S/last_policy_network/last_policy_network/out/b/read:0
Е
7A2S/current_value_network/current_value_network/fc0/w:0<A2S/current_value_network/current_value_network/fc0/w/Assign<A2S/current_value_network/current_value_network/fc0/w/read:0
Е
7A2S/current_value_network/current_value_network/fc0/b:0<A2S/current_value_network/current_value_network/fc0/b/Assign<A2S/current_value_network/current_value_network/fc0/b/read:0
Е
7A2S/current_value_network/current_value_network/fc1/w:0<A2S/current_value_network/current_value_network/fc1/w/Assign<A2S/current_value_network/current_value_network/fc1/w/read:0
Е
7A2S/current_value_network/current_value_network/fc1/b:0<A2S/current_value_network/current_value_network/fc1/b/Assign<A2S/current_value_network/current_value_network/fc1/b/read:0
Е
7A2S/current_value_network/current_value_network/out/w:0<A2S/current_value_network/current_value_network/out/w/Assign<A2S/current_value_network/current_value_network/out/w/read:0
Е
7A2S/current_value_network/current_value_network/out/b:0<A2S/current_value_network/current_value_network/out/b/Assign<A2S/current_value_network/current_value_network/out/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc1/w:06A2S/best_value_network/best_value_network/fc1/w/Assign6A2S/best_value_network/best_value_network/fc1/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc1/b:06A2S/best_value_network/best_value_network/fc1/b/Assign6A2S/best_value_network/best_value_network/fc1/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0

/A2S/current_q_network/current_q_network/fc0/w:04A2S/current_q_network/current_q_network/fc0/w/Assign4A2S/current_q_network/current_q_network/fc0/w/read:0

/A2S/current_q_network/current_q_network/fc0/b:04A2S/current_q_network/current_q_network/fc0/b/Assign4A2S/current_q_network/current_q_network/fc0/b/read:0

/A2S/current_q_network/current_q_network/fc1/w:04A2S/current_q_network/current_q_network/fc1/w/Assign4A2S/current_q_network/current_q_network/fc1/w/read:0

/A2S/current_q_network/current_q_network/fc1/b:04A2S/current_q_network/current_q_network/fc1/b/Assign4A2S/current_q_network/current_q_network/fc1/b/read:0

/A2S/current_q_network/current_q_network/out/w:04A2S/current_q_network/current_q_network/out/w/Assign4A2S/current_q_network/current_q_network/out/w/read:0

/A2S/current_q_network/current_q_network/out/b:04A2S/current_q_network/current_q_network/out/b/Assign4A2S/current_q_network/current_q_network/out/b/read:0

)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0

)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0

)A2S/best_q_network/best_q_network/fc1/w:0.A2S/best_q_network/best_q_network/fc1/w/Assign.A2S/best_q_network/best_q_network/fc1/w/read:0

)A2S/best_q_network/best_q_network/fc1/b:0.A2S/best_q_network/best_q_network/fc1/b/Assign.A2S/best_q_network/best_q_network/fc1/b/read:0

)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0

)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0"0
train_op$
"
A2S/Adam

A2S/Adam_1

A2S/Adam_2"гu
	variablesХuТu
Л
9A2S/current_policy_network/current_policy_network/fc0/w:0>A2S/current_policy_network/current_policy_network/fc0/w/Assign>A2S/current_policy_network/current_policy_network/fc0/w/read:0
Л
9A2S/current_policy_network/current_policy_network/fc0/b:0>A2S/current_policy_network/current_policy_network/fc0/b/Assign>A2S/current_policy_network/current_policy_network/fc0/b/read:0
Л
9A2S/current_policy_network/current_policy_network/fc1/w:0>A2S/current_policy_network/current_policy_network/fc1/w/Assign>A2S/current_policy_network/current_policy_network/fc1/w/read:0
Л
9A2S/current_policy_network/current_policy_network/fc1/b:0>A2S/current_policy_network/current_policy_network/fc1/b/Assign>A2S/current_policy_network/current_policy_network/fc1/b/read:0
Л
9A2S/current_policy_network/current_policy_network/out/w:0>A2S/current_policy_network/current_policy_network/out/w/Assign>A2S/current_policy_network/current_policy_network/out/w/read:0
Л
9A2S/current_policy_network/current_policy_network/out/b:0>A2S/current_policy_network/current_policy_network/out/b/Assign>A2S/current_policy_network/current_policy_network/out/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/w:08A2S/best_policy_network/best_policy_network/fc0/w/Assign8A2S/best_policy_network/best_policy_network/fc0/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc0/b:08A2S/best_policy_network/best_policy_network/fc0/b/Assign8A2S/best_policy_network/best_policy_network/fc0/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc1/w:08A2S/best_policy_network/best_policy_network/fc1/w/Assign8A2S/best_policy_network/best_policy_network/fc1/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/fc1/b:08A2S/best_policy_network/best_policy_network/fc1/b/Assign8A2S/best_policy_network/best_policy_network/fc1/b/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/w:08A2S/best_policy_network/best_policy_network/out/w/Assign8A2S/best_policy_network/best_policy_network/out/w/read:0
Љ
3A2S/best_policy_network/best_policy_network/out/b:08A2S/best_policy_network/best_policy_network/out/b/Assign8A2S/best_policy_network/best_policy_network/out/b/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc0/w:08A2S/last_policy_network/last_policy_network/fc0/w/Assign8A2S/last_policy_network/last_policy_network/fc0/w/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc0/b:08A2S/last_policy_network/last_policy_network/fc0/b/Assign8A2S/last_policy_network/last_policy_network/fc0/b/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc1/w:08A2S/last_policy_network/last_policy_network/fc1/w/Assign8A2S/last_policy_network/last_policy_network/fc1/w/read:0
Љ
3A2S/last_policy_network/last_policy_network/fc1/b:08A2S/last_policy_network/last_policy_network/fc1/b/Assign8A2S/last_policy_network/last_policy_network/fc1/b/read:0
Љ
3A2S/last_policy_network/last_policy_network/out/w:08A2S/last_policy_network/last_policy_network/out/w/Assign8A2S/last_policy_network/last_policy_network/out/w/read:0
Љ
3A2S/last_policy_network/last_policy_network/out/b:08A2S/last_policy_network/last_policy_network/out/b/Assign8A2S/last_policy_network/last_policy_network/out/b/read:0
Е
7A2S/current_value_network/current_value_network/fc0/w:0<A2S/current_value_network/current_value_network/fc0/w/Assign<A2S/current_value_network/current_value_network/fc0/w/read:0
Е
7A2S/current_value_network/current_value_network/fc0/b:0<A2S/current_value_network/current_value_network/fc0/b/Assign<A2S/current_value_network/current_value_network/fc0/b/read:0
Е
7A2S/current_value_network/current_value_network/fc1/w:0<A2S/current_value_network/current_value_network/fc1/w/Assign<A2S/current_value_network/current_value_network/fc1/w/read:0
Е
7A2S/current_value_network/current_value_network/fc1/b:0<A2S/current_value_network/current_value_network/fc1/b/Assign<A2S/current_value_network/current_value_network/fc1/b/read:0
Е
7A2S/current_value_network/current_value_network/out/w:0<A2S/current_value_network/current_value_network/out/w/Assign<A2S/current_value_network/current_value_network/out/w/read:0
Е
7A2S/current_value_network/current_value_network/out/b:0<A2S/current_value_network/current_value_network/out/b/Assign<A2S/current_value_network/current_value_network/out/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/w:06A2S/best_value_network/best_value_network/fc0/w/Assign6A2S/best_value_network/best_value_network/fc0/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc0/b:06A2S/best_value_network/best_value_network/fc0/b/Assign6A2S/best_value_network/best_value_network/fc0/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc1/w:06A2S/best_value_network/best_value_network/fc1/w/Assign6A2S/best_value_network/best_value_network/fc1/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/fc1/b:06A2S/best_value_network/best_value_network/fc1/b/Assign6A2S/best_value_network/best_value_network/fc1/b/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/w:06A2S/best_value_network/best_value_network/out/w/Assign6A2S/best_value_network/best_value_network/out/w/read:0
Ѓ
1A2S/best_value_network/best_value_network/out/b:06A2S/best_value_network/best_value_network/out/b/Assign6A2S/best_value_network/best_value_network/out/b/read:0

/A2S/current_q_network/current_q_network/fc0/w:04A2S/current_q_network/current_q_network/fc0/w/Assign4A2S/current_q_network/current_q_network/fc0/w/read:0

/A2S/current_q_network/current_q_network/fc0/b:04A2S/current_q_network/current_q_network/fc0/b/Assign4A2S/current_q_network/current_q_network/fc0/b/read:0

/A2S/current_q_network/current_q_network/fc1/w:04A2S/current_q_network/current_q_network/fc1/w/Assign4A2S/current_q_network/current_q_network/fc1/w/read:0

/A2S/current_q_network/current_q_network/fc1/b:04A2S/current_q_network/current_q_network/fc1/b/Assign4A2S/current_q_network/current_q_network/fc1/b/read:0

/A2S/current_q_network/current_q_network/out/w:04A2S/current_q_network/current_q_network/out/w/Assign4A2S/current_q_network/current_q_network/out/w/read:0

/A2S/current_q_network/current_q_network/out/b:04A2S/current_q_network/current_q_network/out/b/Assign4A2S/current_q_network/current_q_network/out/b/read:0

)A2S/best_q_network/best_q_network/fc0/w:0.A2S/best_q_network/best_q_network/fc0/w/Assign.A2S/best_q_network/best_q_network/fc0/w/read:0

)A2S/best_q_network/best_q_network/fc0/b:0.A2S/best_q_network/best_q_network/fc0/b/Assign.A2S/best_q_network/best_q_network/fc0/b/read:0

)A2S/best_q_network/best_q_network/fc1/w:0.A2S/best_q_network/best_q_network/fc1/w/Assign.A2S/best_q_network/best_q_network/fc1/w/read:0

)A2S/best_q_network/best_q_network/fc1/b:0.A2S/best_q_network/best_q_network/fc1/b/Assign.A2S/best_q_network/best_q_network/fc1/b/read:0

)A2S/best_q_network/best_q_network/out/w:0.A2S/best_q_network/best_q_network/out/w/Assign.A2S/best_q_network/best_q_network/out/w/read:0

)A2S/best_q_network/best_q_network/out/b:0.A2S/best_q_network/best_q_network/out/b/Assign.A2S/best_q_network/best_q_network/out/b/read:0
C
A2S/beta1_power:0A2S/beta1_power/AssignA2S/beta1_power/read:0
C
A2S/beta2_power:0A2S/beta2_power/AssignA2S/beta2_power/read:0
ж
BA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam/read:0
м
DA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc0/w/Adam_1/read:0
ж
BA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam/read:0
м
DA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc0/b/Adam_1/read:0
ж
BA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam/read:0
м
DA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc1/w/Adam_1/read:0
ж
BA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam:0GA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam/read:0
м
DA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/fc1/b/Adam_1/read:0
ж
BA2S/A2S/current_policy_network/current_policy_network/out/w/Adam:0GA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/out/w/Adam/read:0
м
DA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/out/w/Adam_1/read:0
ж
BA2S/A2S/current_policy_network/current_policy_network/out/b/Adam:0GA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/AssignGA2S/A2S/current_policy_network/current_policy_network/out/b/Adam/read:0
м
DA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1:0IA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/AssignIA2S/A2S/current_policy_network/current_policy_network/out/b/Adam_1/read:0
I
A2S/beta1_power_1:0A2S/beta1_power_1/AssignA2S/beta1_power_1/read:0
I
A2S/beta2_power_1:0A2S/beta2_power_1/AssignA2S/beta2_power_1/read:0
а
@A2S/A2S/current_value_network/current_value_network/fc0/w/Adam:0EA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc0/w/Adam/read:0
ж
BA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc0/w/Adam_1/read:0
а
@A2S/A2S/current_value_network/current_value_network/fc0/b/Adam:0EA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc0/b/Adam/read:0
ж
BA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc0/b/Adam_1/read:0
а
@A2S/A2S/current_value_network/current_value_network/fc1/w/Adam:0EA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc1/w/Adam/read:0
ж
BA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc1/w/Adam_1/read:0
а
@A2S/A2S/current_value_network/current_value_network/fc1/b/Adam:0EA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/AssignEA2S/A2S/current_value_network/current_value_network/fc1/b/Adam/read:0
ж
BA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1:0GA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/fc1/b/Adam_1/read:0
а
@A2S/A2S/current_value_network/current_value_network/out/w/Adam:0EA2S/A2S/current_value_network/current_value_network/out/w/Adam/AssignEA2S/A2S/current_value_network/current_value_network/out/w/Adam/read:0
ж
BA2S/A2S/current_value_network/current_value_network/out/w/Adam_1:0GA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/out/w/Adam_1/read:0
а
@A2S/A2S/current_value_network/current_value_network/out/b/Adam:0EA2S/A2S/current_value_network/current_value_network/out/b/Adam/AssignEA2S/A2S/current_value_network/current_value_network/out/b/Adam/read:0
ж
BA2S/A2S/current_value_network/current_value_network/out/b/Adam_1:0GA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/AssignGA2S/A2S/current_value_network/current_value_network/out/b/Adam_1/read:0
I
A2S/beta1_power_2:0A2S/beta1_power_2/AssignA2S/beta1_power_2/read:0
I
A2S/beta2_power_2:0A2S/beta2_power_2/AssignA2S/beta2_power_2/read:0
И
8A2S/A2S/current_q_network/current_q_network/fc0/w/Adam:0=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc0/w/Adam/read:0
О
:A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc0/w/Adam_1/read:0
И
8A2S/A2S/current_q_network/current_q_network/fc0/b/Adam:0=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc0/b/Adam/read:0
О
:A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc0/b/Adam_1/read:0
И
8A2S/A2S/current_q_network/current_q_network/fc1/w/Adam:0=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc1/w/Adam/read:0
О
:A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc1/w/Adam_1/read:0
И
8A2S/A2S/current_q_network/current_q_network/fc1/b/Adam:0=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/Assign=A2S/A2S/current_q_network/current_q_network/fc1/b/Adam/read:0
О
:A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1:0?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/fc1/b/Adam_1/read:0
И
8A2S/A2S/current_q_network/current_q_network/out/w/Adam:0=A2S/A2S/current_q_network/current_q_network/out/w/Adam/Assign=A2S/A2S/current_q_network/current_q_network/out/w/Adam/read:0
О
:A2S/A2S/current_q_network/current_q_network/out/w/Adam_1:0?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/out/w/Adam_1/read:0
И
8A2S/A2S/current_q_network/current_q_network/out/b/Adam:0=A2S/A2S/current_q_network/current_q_network/out/b/Adam/Assign=A2S/A2S/current_q_network/current_q_network/out/b/Adam/read:0
О
:A2S/A2S/current_q_network/current_q_network/out/b/Adam_1:0?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/Assign?A2S/A2S/current_q_network/current_q_network/out/b/Adam_1/read:0 УB*       Ўѕ	Хa"RXжA*

A2S/average_reward_1-УХAXPё*       Ўѕ	(RXжA%*

A2S/average_reward_1:ЧAЅB3*       Ўѕ	Р*RXжA.*

A2S/average_reward_1ЇТA=Њ*       Ўѕ	^М2RXжAI*

A2S/average_reward_1{EЦAyYЌ*       Ўѕ	 Т9RXжAc*

A2S/average_reward_1PќA>uа*       Ўѕ	|;RXжAj*

A2S/average_reward_1ќ­A=ЄГ|+       УK	ЖнGRXжA*

A2S/average_reward_14/е@7§Ђ+       УK	RRXжAТ*

A2S/average_reward_1ЎM@GЕtж+       УK	8)VRXжAв*

A2S/average_reward_1АшКAGз2ы+       УK	CJYRXжAт*

A2S/average_reward_1GЖAeј+       УK	BN`RXжA*

A2S/average_reward_1Asў(~+       УK	cdRXжA*

A2S/average_reward_1[ђЬAmkфЇ+       УK	/hRXжAА*

A2S/average_reward_1дсAj Ь{+       УK	ндsRXжAк*

A2S/average_reward_1OAeЎ_N+       УK	b}RXжA*

A2S/average_reward_1ЌпCAйk+       УK	FФRXжA*

A2S/average_reward_1наAXќs+       УK	UіRXжAЂ*

A2S/average_reward_1:fРAyрzГ+       УK	 RXжAН*

A2S/average_reward_1rИњA`I&§+       УK	RXжAж*

A2S/average_reward_1З~аAШѕЎ+       УK	Ј$RXжAћ*

A2S/average_reward_1uhЧ@8йЃОw       цДІУ	чрв]XжAћ*i

A2S/klўћ@

A2S/policy_network_lossЭ`ЅО

A2S/value_network_lossЙ&B

A2S/q_network_loss­(BfgН+       УK	жмд]XжA*

A2S/average_reward_1;ЁAфЄ^+       УK	Ц;з]XжA*

A2S/average_reward_1ыП­AhДА+       УK	Ѓ?к]XжA*

A2S/average_reward_1YШA№>+       УK	Gм]XжA *

A2S/average_reward_1ВAA-Йг+       УK	Їп]XжAЊ*

A2S/average_reward_1.pЖAй,+       УK	rZф]XжAМ*

A2S/average_reward_1­ЃA~Л++       УK	Ёёц]XжAХ*

A2S/average_reward_1%rЖAиoXЃ+       УK	Ђь]XжAл*

A2S/average_reward_1є=ІAdN+       УK	П7ё]XжAю*

A2S/average_reward_1TВAzoB+       УK	,ѓ]XжAї*

A2S/average_reward_1ЮЋAЄЋк+       УK	нє]XжAџ*

A2S/average_reward_1а;ЙAяљm+       УK	uGі]XжA*

A2S/average_reward_1V*ЃAўEmB+       УK	щљ]XжA*

A2S/average_reward_1џБAcЮhљ+       УK	њ]XжA*

A2S/average_reward_1хУБA=Ъ8+       УK	цў]XжAЏ*

A2S/average_reward_1=тЭA*3l+       УK	э ^XжAЙ*

A2S/average_reward_1&ЗОArVЇd+       УK	их^XжAТ*

A2S/average_reward_1ЗНAд9o6+       УK	ob^XжAЫ*

A2S/average_reward_1чjОAр§L+       УK	Ѕї^XжAд*

A2S/average_reward_1ЉКAL]7+       УK	J^XжAщ*

A2S/average_reward_1|gФA/ї*+       УK	zD^XжAѓ*

A2S/average_reward_1QЦAОл+       УK	ё^XжAќ*

A2S/average_reward_1ЗMЌA*0ё+       УK	}5^XжA*

A2S/average_reward_1№уАAбЁD+       УK	ч^XжA*

A2S/average_reward_12КAmИ+       УK	zУ^XжA*

A2S/average_reward_1QЉAэМ+       УK	БФ^XжA*

A2S/average_reward_1iАAЦф+       УK	вј^XжAІ*

A2S/average_reward_16тЉA*fзЏ+       УK	>!^XжAЎ*

A2S/average_reward_1eAБAДW+       УK	#^XжAЖ*

A2S/average_reward_1|ДЇAu;+       УK	2 &^XжAП*

A2S/average_reward_1ќpОAДїM+       УK	-$(^XжAШ*

A2S/average_reward_1зйДAн +       УK	Gf*^XжAб*

A2S/average_reward_1їPДA ѕф+       УK	И/^XжAф*

A2S/average_reward_1QуAѕKLZ+       УK	3J2^XжAю*

A2S/average_reward_1DДAЃ(NЫ+       УK	%4^XжAѕ*

A2S/average_reward_1ѕЉAвР+       УK	R8^XжA*

A2S/average_reward_1ЅПAЄТK+       УK	Lџ9^XжA*

A2S/average_reward_1xљБAоk|д+       УK	§К;^XжA*

A2S/average_reward_1јЎA;ЗІ2+       УK	Sq=^XжAЁ*

A2S/average_reward_1~ЌAq+       УK	p?^XжAЊ*

A2S/average_reward_1 ДAиБH+       УK	iqA^XжAГ*

A2S/average_reward_1qГAюЈSУ+       УK	.C^XжAН*

A2S/average_reward_1;}МAї.эн+       УK	>&G^XжAЭ*

A2S/average_reward_1нТA)Ке+       УK	њ K^XжAс*

A2S/average_reward_1GСAЬ§R+       УK	НЮM^XжAъ*

A2S/average_reward_1mЗ­A*CP+       УK	UZR^XжAџ*

A2S/average_reward_1ЅСЯAAєЇ