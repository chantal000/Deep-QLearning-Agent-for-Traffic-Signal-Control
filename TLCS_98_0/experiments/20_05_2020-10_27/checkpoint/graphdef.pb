
”
:main_level/agent/main/online/global_step/Initializer/zerosConst*;
_class1
/-loc:@main_level/agent/main/online/global_step*
value	B	 R *
dtype0	
Ą
(main_level/agent/main/online/global_step
VariableV2"/device:GPU:0*
shape: *
shared_name *;
_class1
/-loc:@main_level/agent/main/online/global_step*
dtype0	*
	container 

/main_level/agent/main/online/global_step/AssignAssign(main_level/agent/main/online/global_step:main_level/agent/main/online/global_step/Initializer/zeros"/device:GPU:0*
use_locking(*
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step*
validate_shape(
ø
-main_level/agent/main/online/global_step/readIdentity(main_level/agent/main/online/global_step"/device:GPU:0*
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
l
3main_level/agent/main/online/Variable/initial_valueConst"/device:GPU:0*
value	B
 Z *
dtype0


%main_level/agent/main/online/Variable
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0
*
	container 

,main_level/agent/main/online/Variable/AssignAssign%main_level/agent/main/online/Variable3main_level/agent/main/online/Variable/initial_value"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/online/Variable*
validate_shape(
Æ
*main_level/agent/main/online/Variable/readIdentity%main_level/agent/main/online/Variable"/device:GPU:0*
T0
*8
_class.
,*loc:@main_level/agent/main/online/Variable
b
(main_level/agent/main/online/PlaceholderPlaceholder"/device:GPU:0*
shape:*
dtype0

ł
#main_level/agent/main/online/AssignAssign%main_level/agent/main/online/Variable(main_level/agent/main/online/Placeholder"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/online/Variable*
validate_shape(

>main_level/agent/main/online/network_0/observation/observationPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
x
<main_level/agent/main/online/network_0/observation/truediv/yConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ū
:main_level/agent/main/online/network_0/observation/truedivRealDiv>main_level/agent/main/online/network_0/observation/observation<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
t
8main_level/agent/main/online/network_0/observation/sub/yConst"/device:GPU:0*
valueB
 *    *
dtype0
Ė
6main_level/agent/main/online/network_0/observation/subSub:main_level/agent/main/online/network_0/observation/truediv8main_level/agent/main/online/network_0/observation/sub/y"/device:GPU:0*
T0
ķ
bmain_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/shapeConst*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
valueB"   @   *
dtype0
ć
`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/minConst*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
valueB
 *0¾*
dtype0
ć
`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/maxConst*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
valueB
 *0>*
dtype0
ä
jmain_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformbmain_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
dtype0*
seed2 

`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/subSub`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/max`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel

`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/mulMuljmain_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniform`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/sub*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel

\main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniformAdd`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/mul`main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel
ś
Amain_level/agent/main/online/network_0/observation/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
dtype0*
	container 

Hmain_level/agent/main/online/network_0/observation/Dense_0/kernel/AssignAssignAmain_level/agent/main/online/network_0/observation/Dense_0/kernel\main_level/agent/main/online/network_0/observation/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
validate_shape(

Fmain_level/agent/main/online/network_0/observation/Dense_0/kernel/readIdentityAmain_level/agent/main/online/network_0/observation/Dense_0/kernel"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel
Ö
Qmain_level/agent/main/online/network_0/observation/Dense_0/bias/Initializer/zerosConst*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
valueB@*    *
dtype0
ņ
?main_level/agent/main/online/network_0/observation/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
dtype0*
	container 
ł
Fmain_level/agent/main/online/network_0/observation/Dense_0/bias/AssignAssign?main_level/agent/main/online/network_0/observation/Dense_0/biasQmain_level/agent/main/online/network_0/observation/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
validate_shape(
ż
Dmain_level/agent/main/online/network_0/observation/Dense_0/bias/readIdentity?main_level/agent/main/online/network_0/observation/Dense_0/bias"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias

Amain_level/agent/main/online/network_0/observation/Dense_0/MatMulMatMul6main_level/agent/main/online/network_0/observation/subFmain_level/agent/main/online/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Bmain_level/agent/main/online/network_0/observation/Dense_0/BiasAddBiasAddAmain_level/agent/main/online/network_0/observation/Dense_0/MatMulDmain_level/agent/main/online/network_0/observation/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
¾
Zmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activationTanhBmain_level/agent/main/online/network_0/observation/Dense_0/BiasAdd"/device:GPU:0*
T0
Õ
Hmain_level/agent/main/online/network_0/observation/Flatten/flatten/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0

Vmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Xmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Xmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
æ
Pmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_sliceStridedSliceHmain_level/agent/main/online/network_0/observation/Flatten/flatten/ShapeVmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stackXmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_1Xmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

Rmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shape/1Const"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
«
Pmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shapePackPmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_sliceRmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shape/1"/device:GPU:0*
T0*

axis *
N
©
Jmain_level/agent/main/online/network_0/observation/Flatten/flatten/ReshapeReshapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activationPmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shape"/device:GPU:0*
T0*
Tshape0

mmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shapeConst*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
valueB"@   @   *
dtype0
ł
kmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/minConst*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]¾*
dtype0
ł
kmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxConst*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]>*
dtype0

umain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformmmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0*
seed2 
¶
kmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/subSubkmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxkmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
Ą
kmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulMulumain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformkmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/sub*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
²
gmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniformAddkmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulkmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel

Lmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 
¶
Smain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/AssignAssignLmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernelgmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
¤
Qmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/readIdentityLmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
ģ
\main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Initializer/zerosConst*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
valueB@*    *
dtype0

Jmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 
„
Qmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/AssignAssignJmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias\main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
validate_shape(

Omain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/readIdentityJmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias
³
Lmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMulMatMulJmain_level/agent/main/online/network_0/observation/Flatten/flatten/ReshapeQmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 
¦
Mmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAddBiasAddLmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMulOmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
Ō
emain_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationTanhMmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAdd"/device:GPU:0*
T0
}
=main_level/agent/main/online/network_0/Variable/initial_valueConst"/device:GPU:0*
valueB*  ?*
dtype0

/main_level/agent/main/online/network_0/Variable
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
	container 
µ
6main_level/agent/main/online/network_0/Variable/AssignAssign/main_level/agent/main/online/network_0/Variable=main_level/agent/main/online/network_0/Variable/initial_value"/device:GPU:0*
use_locking(*
T0*B
_class8
64loc:@main_level/agent/main/online/network_0/Variable*
validate_shape(
Ķ
4main_level/agent/main/online/network_0/Variable/readIdentity/main_level/agent/main/online/network_0/Variable"/device:GPU:0*
T0*B
_class8
64loc:@main_level/agent/main/online/network_0/Variable
A
ConstConst"/device:GPU:0*
valueB
 *  ?*
dtype0

Vmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/initial_valueConst"/device:GPU:0*
valueB
 *  ?*
dtype0
£
Hmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0*
	container 

Omain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/AssignAssignHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalersVmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(

Mmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/readIdentityHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers

Jmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers_1Placeholder"/device:GPU:0*
shape:*
dtype0
ė
-main_level/agent/main/online/network_0/AssignAssignHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalersJmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers_1"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
h
,main_level/agent/main/online/network_0/sub/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ę
*main_level/agent/main/online/network_0/subSub,main_level/agent/main/online/network_0/sub/xMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/read"/device:GPU:0*
T0
Õ
9main_level/agent/main/online/network_0/StopGradient/inputPackemain_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N

3main_level/agent/main/online/network_0/StopGradientStopGradient9main_level/agent/main/online/network_0/StopGradient/input"/device:GPU:0*
T0
Ŗ
*main_level/agent/main/online/network_0/mulMul*main_level/agent/main/online/network_0/sub3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
Ź
.main_level/agent/main/online/network_0/mul_1/yPackemain_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N
Ź
,main_level/agent/main/online/network_0/mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/read.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
£
*main_level/agent/main/online/network_0/addAdd*main_level/agent/main/online/network_0/mul,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0

Jmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
ń
Dmain_level/agent/main/online/network_0/v_values_head_0/strided_sliceStridedSlice*main_level/agent/main/online/network_0/addJmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
ä
Vmain_level/agent/main/online/network_0/v_values_head_0/output/kernel/Initializer/ConstConst*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
valueB@"©G|<jß¾}m9=M=Ä>0¾w_=ü½FÓ>[nX¾¾YĪ6½$ø½ŖQ>Ąz<ZŚ¼§¾k9½ąń=#¾Ia1=ĘY>
B&=$=ÆgQ½%ø:¾ł ö½ĒźÆ<l>>lķN¾āėI=į9F=Æ½ldZ½³lä½~Ńę=³=ģ4Ė=$ ¾Ädå;ļ>¾)SÅ=|=Ū²¹=[%c½ ś=¾čż¼®3@¾}f’=·åĒ=Ć$>æ>ĆH±=½ĪĢ>½Ōo¾'÷½č-Ŗ<8ā½s(A½Z==¤=ŃŪ<¾:)¾*
dtype0

Dmain_level/agent/main/online/network_0/v_values_head_0/output/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
dtype0*
	container 

Kmain_level/agent/main/online/network_0/v_values_head_0/output/kernel/AssignAssignDmain_level/agent/main/online/network_0/v_values_head_0/output/kernelVmain_level/agent/main/online/network_0/v_values_head_0/output/kernel/Initializer/Const"/device:GPU:0*
use_locking(*
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
validate_shape(

Imain_level/agent/main/online/network_0/v_values_head_0/output/kernel/readIdentityDmain_level/agent/main/online/network_0/v_values_head_0/output/kernel"/device:GPU:0*
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel
Ü
Tmain_level/agent/main/online/network_0/v_values_head_0/output/bias/Initializer/zerosConst*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
valueB*    *
dtype0
ų
Bmain_level/agent/main/online/network_0/v_values_head_0/output/bias
VariableV2"/device:GPU:0*
shape:*
shared_name *U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
dtype0*
	container 

Imain_level/agent/main/online/network_0/v_values_head_0/output/bias/AssignAssignBmain_level/agent/main/online/network_0/v_values_head_0/output/biasTmain_level/agent/main/online/network_0/v_values_head_0/output/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
validate_shape(

Gmain_level/agent/main/online/network_0/v_values_head_0/output/bias/readIdentityBmain_level/agent/main/online/network_0/v_values_head_0/output/bias"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias

Dmain_level/agent/main/online/network_0/v_values_head_0/output/MatMulMatMulDmain_level/agent/main/online/network_0/v_values_head_0/strided_sliceImain_level/agent/main/online/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Emain_level/agent/main/online/network_0/v_values_head_0/output/BiasAddBiasAddDmain_level/agent/main/online/network_0/v_values_head_0/output/MatMulGmain_level/agent/main/online/network_0/v_values_head_0/output/bias/read"/device:GPU:0*
T0*
data_formatNHWC

Mmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0_targetPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
”
Xmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0_importance_weightPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1/stackConst"/device:GPU:0*
valueB: *
dtype0

Nmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Nmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1/stack_2Const"/device:GPU:0*
valueB:*
dtype0

Fmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1StridedSlice4main_level/agent/main/online/network_0/Variable/readLmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1/stackNmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1/stack_1Nmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
ū
:main_level/agent/main/online/network_0/v_values_head_0/mulMulFmain_level/agent/main/online/network_0/v_values_head_0/strided_slice_1Xmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0_importance_weight"/device:GPU:0*
T0

Xmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifferenceSquaredDifferenceEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAddMmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0_target"/device:GPU:0*
T0

cmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/weightsConst"/device:GPU:0*
valueB
 *  ?*
dtype0
”
imain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/weights/shapeConst"/device:GPU:0*
valueB *
dtype0
”
hmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/weights/rankConst"/device:GPU:0*
value	B : *
dtype0
ó
hmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/values/shapeShapeXmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference"/device:GPU:0*
T0*
out_type0
 
gmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/values/rankConst"/device:GPU:0*
value	B :*
dtype0

wmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/static_scalar_check_successNoOp"/device:GPU:0

Mmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Cast/xConstx^main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/static_scalar_check_success"/device:GPU:0*
valueB
 *  ?*
dtype0

Jmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/MulMulXmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifferenceMmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Cast/x"/device:GPU:0*
T0

Lmain_level/agent/main/online/network_0/v_values_head_0/Sum/reduction_indicesConst"/device:GPU:0*
valueB:*
dtype0

:main_level/agent/main/online/network_0/v_values_head_0/SumSumJmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/MulLmain_level/agent/main/online/network_0/v_values_head_0/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ó
<main_level/agent/main/online/network_0/v_values_head_0/mul_1Mul:main_level/agent/main/online/network_0/v_values_head_0/mul:main_level/agent/main/online/network_0/v_values_head_0/Sum"/device:GPU:0*
T0

<main_level/agent/main/online/network_0/v_values_head_0/ConstConst"/device:GPU:0*
valueB"       *
dtype0
ō
;main_level/agent/main/online/network_0/v_values_head_0/MeanMean<main_level/agent/main/online/network_0/v_values_head_0/mul_1<main_level/agent/main/online/network_0/v_values_head_0/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
x
<main_level/agent/main/online/network_1/observation/truediv/yConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ū
:main_level/agent/main/online/network_1/observation/truedivRealDiv>main_level/agent/main/online/network_0/observation/observation<main_level/agent/main/online/network_1/observation/truediv/y"/device:GPU:0*
T0
t
8main_level/agent/main/online/network_1/observation/sub/yConst"/device:GPU:0*
valueB
 *    *
dtype0
Ė
6main_level/agent/main/online/network_1/observation/subSub:main_level/agent/main/online/network_1/observation/truediv8main_level/agent/main/online/network_1/observation/sub/y"/device:GPU:0*
T0
ķ
bmain_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/shapeConst*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
valueB"   @   *
dtype0
ć
`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/minConst*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
valueB
 *0¾*
dtype0
ć
`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/maxConst*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
valueB
 *0>*
dtype0
ä
jmain_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformbmain_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
dtype0*
seed2 

`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/subSub`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/max`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel

`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/mulMuljmain_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniform`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/sub*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel

\main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniformAdd`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/mul`main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel
ś
Amain_level/agent/main/online/network_1/observation/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
dtype0*
	container 

Hmain_level/agent/main/online/network_1/observation/Dense_0/kernel/AssignAssignAmain_level/agent/main/online/network_1/observation/Dense_0/kernel\main_level/agent/main/online/network_1/observation/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
validate_shape(

Fmain_level/agent/main/online/network_1/observation/Dense_0/kernel/readIdentityAmain_level/agent/main/online/network_1/observation/Dense_0/kernel"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel
Ö
Qmain_level/agent/main/online/network_1/observation/Dense_0/bias/Initializer/zerosConst*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
valueB@*    *
dtype0
ņ
?main_level/agent/main/online/network_1/observation/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
dtype0*
	container 
ł
Fmain_level/agent/main/online/network_1/observation/Dense_0/bias/AssignAssign?main_level/agent/main/online/network_1/observation/Dense_0/biasQmain_level/agent/main/online/network_1/observation/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
validate_shape(
ż
Dmain_level/agent/main/online/network_1/observation/Dense_0/bias/readIdentity?main_level/agent/main/online/network_1/observation/Dense_0/bias"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias

Amain_level/agent/main/online/network_1/observation/Dense_0/MatMulMatMul6main_level/agent/main/online/network_1/observation/subFmain_level/agent/main/online/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Bmain_level/agent/main/online/network_1/observation/Dense_0/BiasAddBiasAddAmain_level/agent/main/online/network_1/observation/Dense_0/MatMulDmain_level/agent/main/online/network_1/observation/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
¾
Zmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activationTanhBmain_level/agent/main/online/network_1/observation/Dense_0/BiasAdd"/device:GPU:0*
T0
Õ
Hmain_level/agent/main/online/network_1/observation/Flatten/flatten/ShapeShapeZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0

Vmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Xmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Xmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
æ
Pmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_sliceStridedSliceHmain_level/agent/main/online/network_1/observation/Flatten/flatten/ShapeVmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_slice/stackXmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_slice/stack_1Xmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

Rmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape/shape/1Const"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
«
Pmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape/shapePackPmain_level/agent/main/online/network_1/observation/Flatten/flatten/strided_sliceRmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape/shape/1"/device:GPU:0*
T0*

axis *
N
©
Jmain_level/agent/main/online/network_1/observation/Flatten/flatten/ReshapeReshapeZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activationPmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape/shape"/device:GPU:0*
T0*
Tshape0

mmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shapeConst*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
valueB"@   @   *
dtype0
ł
kmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/minConst*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]¾*
dtype0
ł
kmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxConst*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]>*
dtype0

umain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformmmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0*
seed2 
¶
kmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/subSubkmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxkmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
Ą
kmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulMulumain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformkmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/sub*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
²
gmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniformAddkmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulkmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel

Lmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 
¶
Smain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/AssignAssignLmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernelgmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
¤
Qmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/readIdentityLmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
ģ
\main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Initializer/zerosConst*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
valueB@*    *
dtype0

Jmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 
„
Qmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/AssignAssignJmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias\main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
validate_shape(

Omain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/readIdentityJmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias
³
Lmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMulMatMulJmain_level/agent/main/online/network_1/observation/Flatten/flatten/ReshapeQmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 
¦
Mmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAddBiasAddLmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMulOmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
Ō
emain_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationTanhMmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAdd"/device:GPU:0*
T0
}
=main_level/agent/main/online/network_1/Variable/initial_valueConst"/device:GPU:0*
valueB*  ?*
dtype0

/main_level/agent/main/online/network_1/Variable
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
	container 
µ
6main_level/agent/main/online/network_1/Variable/AssignAssign/main_level/agent/main/online/network_1/Variable=main_level/agent/main/online/network_1/Variable/initial_value"/device:GPU:0*
use_locking(*
T0*B
_class8
64loc:@main_level/agent/main/online/network_1/Variable*
validate_shape(
Ķ
4main_level/agent/main/online/network_1/Variable/readIdentity/main_level/agent/main/online/network_1/Variable"/device:GPU:0*
T0*B
_class8
64loc:@main_level/agent/main/online/network_1/Variable
C
Const_1Const"/device:GPU:0*
valueB
 *  ?*
dtype0

Vmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/initial_valueConst"/device:GPU:0*
valueB
 *  ?*
dtype0
£
Hmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0*
	container 

Omain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/AssignAssignHmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalersVmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
validate_shape(

Mmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/readIdentityHmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers

Hmain_level/agent/main/online/network_1/gradients_from_head_1-0_rescalersPlaceholder"/device:GPU:0*
shape:*
dtype0
é
-main_level/agent/main/online/network_1/AssignAssignHmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalersHmain_level/agent/main/online/network_1/gradients_from_head_1-0_rescalers"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
validate_shape(
h
,main_level/agent/main/online/network_1/sub/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ę
*main_level/agent/main/online/network_1/subSub,main_level/agent/main/online/network_1/sub/xMmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/read"/device:GPU:0*
T0
Õ
9main_level/agent/main/online/network_1/StopGradient/inputPackemain_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N

3main_level/agent/main/online/network_1/StopGradientStopGradient9main_level/agent/main/online/network_1/StopGradient/input"/device:GPU:0*
T0
Ŗ
*main_level/agent/main/online/network_1/mulMul*main_level/agent/main/online/network_1/sub3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0
Ź
.main_level/agent/main/online/network_1/mul_1/yPackemain_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N
Ź
,main_level/agent/main/online/network_1/mul_1MulMmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/read.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0
£
*main_level/agent/main/online/network_1/addAdd*main_level/agent/main/online/network_1/mul,main_level/agent/main/online/network_1/mul_1"/device:GPU:0*
T0

Emain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Gmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Gmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
Ż
?main_level/agent/main/online/network_1/ppo_head_0/strided_sliceStridedSlice*main_level/agent/main/online/network_1/addEmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
~
9main_level/agent/main/online/network_1/ppo_head_0/actionsPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

Amain_level/agent/main/online/network_1/ppo_head_0/old_policy_meanPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

@main_level/agent/main/online/network_1/ppo_head_0/old_policy_stdPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
ļ
cmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/shapeConst*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
valueB"@      *
dtype0
å
amain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/minConst*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
valueB
 *²_¾*
dtype0
å
amain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/maxConst*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
valueB
 *²_>*
dtype0
ē
kmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniformcmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/shape*

seed *
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
dtype0*
seed2 

amain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/subSubamain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/maxamain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/min*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel

amain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/mulMulkmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/RandomUniformamain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/sub*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel

]main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniformAddamain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/mulamain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/min*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel
ü
Bmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
dtype0*
	container 

Imain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/AssignAssignBmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel]main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
validate_shape(

Gmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/readIdentityBmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel
Ų
Rmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Initializer/zerosConst*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
valueB*    *
dtype0
ō
@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias
VariableV2"/device:GPU:0*
shape:*
shared_name *S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
dtype0*
	container 
ż
Gmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/AssignAssign@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/biasRmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
validate_shape(

Emain_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/readIdentity@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias"/device:GPU:0*
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias

Bmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMulMatMul?main_level/agent/main/online/network_1/ppo_head_0/strided_sliceGmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Cmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAddBiasAddBmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMulEmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/read"/device:GPU:0*
T0*
data_formatNHWC
 
8main_level/agent/main/online/network_1/ppo_head_0/policySoftmaxCmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAdd"/device:GPU:0*
T0
”
Hmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/LogLog8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0

Hmain_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_rankConst"/device:GPU:0*
value	B :*
dtype0
Å
Jmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits_shapeShapeHmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0*
out_type0

Hmain_level/agent/main/online/network_1/ppo_head_0/Categorical/event_sizeConst"/device:GPU:0*
value	B :*
dtype0

]main_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
„
_main_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

_main_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
Ż
Wmain_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_sliceStridedSliceJmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits_shape]main_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_main_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_1_main_level/agent/main/online/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 
¬
Jmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits/LogLogAmain_level/agent/main/online/network_1/ppo_head_0/old_policy_mean"/device:GPU:0*
T0

Jmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_rankConst"/device:GPU:0*
value	B :*
dtype0
É
Lmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits_shapeShapeJmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits/Log"/device:GPU:0*
T0*
out_type0

Jmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/event_sizeConst"/device:GPU:0*
value	B :*
dtype0

_main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
§
amain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

amain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
ē
Ymain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_sliceStridedSliceLmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits_shape_main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stackamain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_1amain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

Zmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stackConst"/device:GPU:0*
valueB"        *
dtype0
 
\main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_1Const"/device:GPU:0*
valueB"        *
dtype0
 
\main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_2Const"/device:GPU:0*
valueB"      *
dtype0
Ą
Tmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_sliceStridedSlice9main_level/agent/main/online/network_1/ppo_head_0/actionsZmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack\main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_1\main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask*
new_axis_mask*
end_mask 
Ż
Vmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like/ShapeShapeTmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice"/device:GPU:0*
T0*
out_type0

Vmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like/ConstConst"/device:GPU:0*
valueB
 *  ?*
dtype0
²
Pmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_likeFillVmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like/ShapeVmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like/Const"/device:GPU:0*
T0*

index_type0

Jmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mulMulHmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/LogPmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like"/device:GPU:0*
T0
É
Lmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ShapeShapeJmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul"/device:GPU:0*
T0*
out_type0

\main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stackConst"/device:GPU:0*
valueB: *
dtype0
¤
^main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

^main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_2Const"/device:GPU:0*
valueB:*
dtype0
Ū
Vmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1StridedSliceLmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/Shape\main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack^main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_1^main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

Qmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones/ConstConst"/device:GPU:0*
value	B :*
dtype0
Ø
Kmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/onesFillVmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1Qmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones/Const"/device:GPU:0*
T0*

index_type0
ó
Lmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_1Mul9main_level/agent/main/online/network_1/ppo_head_0/actionsKmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones"/device:GPU:0*
T0
ļ
pmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeLmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_1"/device:GPU:0*
T0*
out_type0
÷
main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsJmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mulLmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_1"/device:GPU:0*
T0*
Tlabels0
ś
Jmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/NegNegmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"/device:GPU:0*
T0
 
\main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stackConst"/device:GPU:0*
valueB"        *
dtype0
¢
^main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_1Const"/device:GPU:0*
valueB"        *
dtype0
¢
^main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_2Const"/device:GPU:0*
valueB"      *
dtype0
Č
Vmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_sliceStridedSlice9main_level/agent/main/online/network_1/ppo_head_0/actions\main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack^main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_1^main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask*
new_axis_mask*
end_mask 
į
Xmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/ShapeShapeVmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice"/device:GPU:0*
T0*
out_type0

Xmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/ConstConst"/device:GPU:0*
valueB
 *  ?*
dtype0
ø
Rmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones_likeFillXmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/ShapeXmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/Const"/device:GPU:0*
T0*

index_type0

Lmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/mulMulJmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits/LogRmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones_like"/device:GPU:0*
T0
Ķ
Nmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ShapeShapeLmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/mul"/device:GPU:0*
T0*
out_type0

^main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stackConst"/device:GPU:0*
valueB: *
dtype0
¦
`main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

`main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_2Const"/device:GPU:0*
valueB:*
dtype0
å
Xmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1StridedSliceNmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/Shape^main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack`main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_1`main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

Smain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones/ConstConst"/device:GPU:0*
value	B :*
dtype0
®
Mmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/onesFillXmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1Smain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones/Const"/device:GPU:0*
T0*

index_type0
÷
Nmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/mul_1Mul9main_level/agent/main/online/network_1/ppo_head_0/actionsMmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/ones"/device:GPU:0*
T0
ó
rmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeNmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/mul_1"/device:GPU:0*
T0*
out_type0
ż
main_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsLmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/mulNmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/mul_1"/device:GPU:0*
T0*
Tlabels0
ž
Lmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/NegNegmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"/device:GPU:0*
T0
Ą
Pmain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/LogSoftmax
LogSoftmaxHmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0
ō
Imain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/mulMulPmain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/LogSoftmax8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0

[main_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
­
Imain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/SumSumImain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/mul[main_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
³
Imain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/NegNegImain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/Sum"/device:GPU:0*
T0
t
7main_level/agent/main/online/network_1/ppo_head_0/ConstConst"/device:GPU:0*
valueB: *
dtype0
÷
6main_level/agent/main/online/network_1/ppo_head_0/MeanMeanImain_level/agent/main/online/network_1/ppo_head_0/Categorical/entropy/Neg7main_level/agent/main/online/network_1/ppo_head_0/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ł
gmain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmax
LogSoftmaxJmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits/Log"/device:GPU:0*
T0
Ł
imain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmax_1
LogSoftmaxHmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0
Ó
`main_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/subSubgmain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmaximain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmax_1"/device:GPU:0*
T0
Ó
dmain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/SoftmaxSoftmaxJmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/logits/Log"/device:GPU:0*
T0
Ē
`main_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/mulMuldmain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Softmax`main_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/sub"/device:GPU:0*
T0
“
rmain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
ņ
`main_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/SumSum`main_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/mulrmain_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
v
9main_level/agent/main/online/network_1/ppo_head_0/Const_1Const"/device:GPU:0*
valueB: *
dtype0

8main_level/agent/main/online/network_1/ppo_head_0/Mean_1Mean`main_level/agent/main/online/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Sum9main_level/agent/main/online/network_1/ppo_head_0/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

<main_level/agent/main/online/network_1/ppo_head_0/advantagesPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
ī
5main_level/agent/main/online/network_1/ppo_head_0/subSubJmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/NegLmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/Neg"/device:GPU:0*
T0

5main_level/agent/main/online/network_1/ppo_head_0/ExpExp5main_level/agent/main/online/network_1/ppo_head_0/sub"/device:GPU:0*
T0
u
=main_level/agent/main/online/network_1/ppo_head_0/PlaceholderPlaceholder"/device:GPU:0*
shape: *
dtype0
s
7main_level/agent/main/online/network_1/ppo_head_0/mul/xConst"/device:GPU:0*
valueB
 *ĶĢL>*
dtype0
Ģ
5main_level/agent/main/online/network_1/ppo_head_0/mulMul7main_level/agent/main/online/network_1/ppo_head_0/mul/x=main_level/agent/main/online/network_1/ppo_head_0/Placeholder"/device:GPU:0*
T0
s
7main_level/agent/main/online/network_1/ppo_head_0/add/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ä
5main_level/agent/main/online/network_1/ppo_head_0/addAdd7main_level/agent/main/online/network_1/ppo_head_0/add/x5main_level/agent/main/online/network_1/ppo_head_0/mul"/device:GPU:0*
T0
u
9main_level/agent/main/online/network_1/ppo_head_0/mul_1/xConst"/device:GPU:0*
valueB
 *ĶĢL>*
dtype0
Š
7main_level/agent/main/online/network_1/ppo_head_0/mul_1Mul9main_level/agent/main/online/network_1/ppo_head_0/mul_1/x=main_level/agent/main/online/network_1/ppo_head_0/Placeholder"/device:GPU:0*
T0
u
9main_level/agent/main/online/network_1/ppo_head_0/sub_1/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ź
7main_level/agent/main/online/network_1/ppo_head_0/sub_1Sub9main_level/agent/main/online/network_1/ppo_head_0/sub_1/x7main_level/agent/main/online/network_1/ppo_head_0/mul_1"/device:GPU:0*
T0
Ų
Gmain_level/agent/main/online/network_1/ppo_head_0/clip_by_value/MinimumMinimum5main_level/agent/main/online/network_1/ppo_head_0/Exp5main_level/agent/main/online/network_1/ppo_head_0/add"/device:GPU:0*
T0
ä
?main_level/agent/main/online/network_1/ppo_head_0/clip_by_valueMaximumGmain_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum7main_level/agent/main/online/network_1/ppo_head_0/sub_1"/device:GPU:0*
T0
Ė
7main_level/agent/main/online/network_1/ppo_head_0/mul_2Mul5main_level/agent/main/online/network_1/ppo_head_0/Exp<main_level/agent/main/online/network_1/ppo_head_0/advantages"/device:GPU:0*
T0
Õ
7main_level/agent/main/online/network_1/ppo_head_0/mul_3Mul?main_level/agent/main/online/network_1/ppo_head_0/clip_by_value<main_level/agent/main/online/network_1/ppo_head_0/advantages"/device:GPU:0*
T0
Ī
9main_level/agent/main/online/network_1/ppo_head_0/MinimumMinimum7main_level/agent/main/online/network_1/ppo_head_0/mul_27main_level/agent/main/online/network_1/ppo_head_0/mul_3"/device:GPU:0*
T0
v
9main_level/agent/main/online/network_1/ppo_head_0/Const_2Const"/device:GPU:0*
valueB: *
dtype0
ė
8main_level/agent/main/online/network_1/ppo_head_0/Mean_2Mean9main_level/agent/main/online/network_1/ppo_head_0/Minimum9main_level/agent/main/online/network_1/ppo_head_0/Const_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

5main_level/agent/main/online/network_1/ppo_head_0/NegNeg8main_level/agent/main/online/network_1/ppo_head_0/Mean_2"/device:GPU:0*
T0

Nmain_level/agent/main/online/network_1/ppo_head_0/ppo_head_0_importance_weightPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
Ń
(main_level/agent/main/online/Rank/packedPack;main_level/agent/main/online/network_0/v_values_head_0/Mean5main_level/agent/main/online/network_1/ppo_head_0/Neg"/device:GPU:0*
T0*

axis *
N
Z
!main_level/agent/main/online/RankConst"/device:GPU:0*
value	B :*
dtype0
a
(main_level/agent/main/online/range/startConst"/device:GPU:0*
value	B : *
dtype0
a
(main_level/agent/main/online/range/deltaConst"/device:GPU:0*
value	B :*
dtype0
½
"main_level/agent/main/online/rangeRange(main_level/agent/main/online/range/start!main_level/agent/main/online/Rank(main_level/agent/main/online/range/delta"/device:GPU:0*

Tidx0
Ļ
&main_level/agent/main/online/Sum/inputPack;main_level/agent/main/online/network_0/v_values_head_0/Mean5main_level/agent/main/online/network_1/ppo_head_0/Neg"/device:GPU:0*
T0*

axis *
N
Ø
 main_level/agent/main/online/SumSum&main_level/agent/main/online/Sum/input"main_level/agent/main/online/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
e
%main_level/agent/main/online/0_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
a
%main_level/agent/main/online/1_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
e
%main_level/agent/main/online/2_holderPlaceholder"/device:GPU:0*
shape
:@@*
dtype0
a
%main_level/agent/main/online/3_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
]
%main_level/agent/main/online/4_holderPlaceholder"/device:GPU:0*
shape: *
dtype0
e
%main_level/agent/main/online/5_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
a
%main_level/agent/main/online/6_holderPlaceholder"/device:GPU:0*
shape:*
dtype0
e
%main_level/agent/main/online/7_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
a
%main_level/agent/main/online/8_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
e
%main_level/agent/main/online/9_holderPlaceholder"/device:GPU:0*
shape
:@@*
dtype0
b
&main_level/agent/main/online/10_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
^
&main_level/agent/main/online/11_holderPlaceholder"/device:GPU:0*
shape: *
dtype0
f
&main_level/agent/main/online/12_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
b
&main_level/agent/main/online/13_holderPlaceholder"/device:GPU:0*
shape:*
dtype0
°
%main_level/agent/main/online/Assign_1AssignAmain_level/agent/main/online/network_0/observation/Dense_0/kernel%main_level/agent/main/online/0_holder"/device:GPU:0*
use_locking( *
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
validate_shape(
¬
%main_level/agent/main/online/Assign_2Assign?main_level/agent/main/online/network_0/observation/Dense_0/bias%main_level/agent/main/online/1_holder"/device:GPU:0*
use_locking( *
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
validate_shape(
Ę
%main_level/agent/main/online/Assign_3AssignLmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel%main_level/agent/main/online/2_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
Ā
%main_level/agent/main/online/Assign_4AssignJmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias%main_level/agent/main/online/3_holder"/device:GPU:0*
use_locking( *
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
validate_shape(
¾
%main_level/agent/main/online/Assign_5AssignHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers%main_level/agent/main/online/4_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
¶
%main_level/agent/main/online/Assign_6AssignDmain_level/agent/main/online/network_0/v_values_head_0/output/kernel%main_level/agent/main/online/5_holder"/device:GPU:0*
use_locking( *
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
validate_shape(
²
%main_level/agent/main/online/Assign_7AssignBmain_level/agent/main/online/network_0/v_values_head_0/output/bias%main_level/agent/main/online/6_holder"/device:GPU:0*
use_locking( *
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
validate_shape(
°
%main_level/agent/main/online/Assign_8AssignAmain_level/agent/main/online/network_1/observation/Dense_0/kernel%main_level/agent/main/online/7_holder"/device:GPU:0*
use_locking( *
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
validate_shape(
¬
%main_level/agent/main/online/Assign_9Assign?main_level/agent/main/online/network_1/observation/Dense_0/bias%main_level/agent/main/online/8_holder"/device:GPU:0*
use_locking( *
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
validate_shape(
Ē
&main_level/agent/main/online/Assign_10AssignLmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel%main_level/agent/main/online/9_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
Ä
&main_level/agent/main/online/Assign_11AssignJmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias&main_level/agent/main/online/10_holder"/device:GPU:0*
use_locking( *
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
validate_shape(
Ą
&main_level/agent/main/online/Assign_12AssignHmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers&main_level/agent/main/online/11_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
validate_shape(
“
&main_level/agent/main/online/Assign_13AssignBmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel&main_level/agent/main/online/12_holder"/device:GPU:0*
use_locking( *
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
validate_shape(
°
&main_level/agent/main/online/Assign_14Assign@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias&main_level/agent/main/online/13_holder"/device:GPU:0*
use_locking( *
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
validate_shape(
d
,main_level/agent/main/online/gradients/ShapeConst"/device:GPU:0*
valueB *
dtype0
l
0main_level/agent/main/online/gradients/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
½
+main_level/agent/main/online/gradients/FillFill,main_level/agent/main/online/gradients/Shape0main_level/agent/main/online/gradients/grad_ys_0"/device:GPU:0*
T0*

index_type0

Zmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0

Tmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/ReshapeReshape+main_level/agent/main/online/gradients/FillZmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0

Rmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/ConstConst"/device:GPU:0*
valueB:*
dtype0
­
Qmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/TileTileTmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/ReshapeRmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Const"/device:GPU:0*

Tmultiples0*
T0
ę
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum/input_grad/unstackUnpackQmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Tile"/device:GPU:0*
T0*	
num*

axis 
¹
umain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Reshape/shapeConst"/device:GPU:0*
valueB"      *
dtype0
ó
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum/input_grad/unstackumain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0
Ü
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/ShapeShape<main_level/agent/main/online/network_0/v_values_head_0/mul_1"/device:GPU:0*
T0*
out_type0
ž
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/TileTileomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Reshapemmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Shape"/device:GPU:0*

Tmultiples0*
T0
Ž
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Shape_1Shape<main_level/agent/main/online/network_0/v_values_head_0/mul_1"/device:GPU:0*
T0*
out_type0
§
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Shape_2Const"/device:GPU:0*
valueB *
dtype0
Ŗ
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0

lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/ProdProdomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Shape_1mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
¬
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0

nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Prod_1Prodomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Shape_2omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ŗ
qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Maximum/yConst"/device:GPU:0*
value	B :*
dtype0
õ
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/MaximumMaximumnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Prod_1qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Maximum/y"/device:GPU:0*
T0
ó
pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/floordivFloorDivlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Prodomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Maximum"/device:GPU:0*
T0

lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/CastCastpmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/floordiv"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
ī
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/truedivRealDivlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Tilelmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/Cast"/device:GPU:0*
T0
ā
emain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Neg_grad/NegNeg\main_level/agent/main/online/gradients/main_level/agent/main/online/Sum/input_grad/unstack:1"/device:GPU:0*
T0
Ū
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/ShapeShape:main_level/agent/main/online/network_0/v_values_head_0/mul"/device:GPU:0*
T0*
out_type0
Ż
pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Shape_1Shape:main_level/agent/main/online/network_0/v_values_head_0/Sum"/device:GPU:0*
T0*
out_type0

~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Shapepmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
ø
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/MulMulomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/truediv:main_level/agent/main/online/network_0/v_values_head_0/Sum"/device:GPU:0*
T0

lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/SumSumlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Mul~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
’
pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/ReshapeReshapelmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Sumnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
ŗ
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Mul_1Mul:main_level/agent/main/online/network_0/v_values_head_0/mulomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Mean_grad/truediv"/device:GPU:0*
T0

nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Sum_1Sumnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Mul_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

rmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Reshape_1Reshapenmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Sum_1pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Æ
rmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0
ų
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/ReshapeReshapeemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Neg_grad/Negrmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0
Ö
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/ShapeShape9main_level/agent/main/online/network_1/ppo_head_0/Minimum"/device:GPU:0*
T0*
out_type0
õ
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/TileTilelmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Reshapejmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Shape"/device:GPU:0*

Tmultiples0*
T0
Ų
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Shape_1Shape9main_level/agent/main/online/network_1/ppo_head_0/Minimum"/device:GPU:0*
T0*
out_type0
¤
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Shape_2Const"/device:GPU:0*
valueB *
dtype0
§
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0

imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/ProdProdlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Shape_1jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
©
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0

kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Prod_1Prodlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Shape_2lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
§
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Maximum/yConst"/device:GPU:0*
value	B :*
dtype0
ģ
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/MaximumMaximumkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Prod_1nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Maximum/y"/device:GPU:0*
T0
ź
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/floordivFloorDivimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Prodlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Maximum"/device:GPU:0*
T0

imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/CastCastmmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/floordiv"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
å
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/truedivRealDivimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Tileimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/Cast"/device:GPU:0*
T0
é
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/ShapeShapeJmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul"/device:GPU:0*
T0*
out_type0
„
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/SizeConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0
Å
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/addAddLmain_level/agent/main/online/network_0/v_values_head_0/Sum/reduction_indiceskmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Size"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape
č
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/modFloorModjmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/addkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Size"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape
¬
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape_1Const"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
valueB:*
dtype0
¬
rmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/range/startConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
value	B : *
dtype0
¬
rmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/range/deltaConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0
ę
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/rangeRangermain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/range/startkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Sizermain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/range/delta"/device:GPU:0*

Tidx0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape
«
qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Fill/valueConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0

kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/FillFillnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape_1qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Fill/value"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*

index_type0
Ü
tmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/DynamicStitchDynamicStitchlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/rangejmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/modlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shapekmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Fill"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
N
Ŗ
pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Maximum/yConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0
ś
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/MaximumMaximumtmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/DynamicStitchpmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Maximum/y"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape
ņ
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/floordivFloorDivlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shapenmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Maximum"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Shape

nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/ReshapeReshapermain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/mul_1_grad/Reshape_1tmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0
ž
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/TileTilenmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Reshapeomain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
Õ
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/ShapeShape7main_level/agent/main/online/network_1/ppo_head_0/mul_2"/device:GPU:0*
T0*
out_type0
×
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shape_1Shape7main_level/agent/main/online/network_1/ppo_head_0/mul_3"/device:GPU:0*
T0*
out_type0

mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shape_2Shapelmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/truediv"/device:GPU:0*
T0*
out_type0
­
qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/zeros/ConstConst"/device:GPU:0*
valueB
 *    *
dtype0
’
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/zerosFillmmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shape_2qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/zeros/Const"/device:GPU:0*
T0*

index_type0

omain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/LessEqual	LessEqual7main_level/agent/main/online/network_1/ppo_head_0/mul_27main_level/agent/main/online/network_1/ppo_head_0/mul_3"/device:GPU:0*
T0

{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgskmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shapemmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shape_1"/device:GPU:0*
T0
Ś
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/SelectSelectomain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/LessEquallmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/truedivkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/zeros"/device:GPU:0*
T0

imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/SumSumlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Select{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ö
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/ReshapeReshapeimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Sumkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ü
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Select_1Selectomain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/LessEqualkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/zeroslmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Mean_2_grad/truediv"/device:GPU:0*
T0

kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Sum_1Sumnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Select_1}main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ü
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Reshape_1Reshapekmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Sum_1mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/ShapeShapeXmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference"/device:GPU:0*
T0*
out_type0
¶
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
¼
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape_1"/device:GPU:0*
T0
Õ
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/MulMulkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/TileMmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Cast/x"/device:GPU:0*
T0
Į
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/SumSumzmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Mulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
©
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/ReshapeReshapezmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Sum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
ā
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Mul_1MulXmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifferencekmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/Sum_grad/Tile"/device:GPU:0*
T0
Ē
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Sum_1Sum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Mul_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
°
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape_1Reshape|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Sum_1~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ń
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/ShapeShape5main_level/agent/main/online/network_1/ppo_head_0/Exp"/device:GPU:0*
T0*
out_type0
Ś
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Shape_1Shape<main_level/agent/main/online/network_1/ppo_head_0/advantages"/device:GPU:0*
T0*
out_type0

ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Shapekmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Shape_1"/device:GPU:0*
T0
³
gmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/MulMulmmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Reshape<main_level/agent/main/online/network_1/ppo_head_0/advantages"/device:GPU:0*
T0

gmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/SumSumgmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Mulymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
š
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/ReshapeReshapegmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Sumimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
®
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Mul_1Mul5main_level/agent/main/online/network_1/ppo_head_0/Expmmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Reshape"/device:GPU:0*
T0

imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Sum_1Sumimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Mul_1{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ö
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Reshape_1Reshapeimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Sum_1kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ū
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/ShapeShape?main_level/agent/main/online/network_1/ppo_head_0/clip_by_value"/device:GPU:0*
T0*
out_type0
Ś
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Shape_1Shape<main_level/agent/main/online/network_1/ppo_head_0/advantages"/device:GPU:0*
T0*
out_type0

ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Shapekmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Shape_1"/device:GPU:0*
T0
µ
gmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/MulMulomain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Reshape_1<main_level/agent/main/online/network_1/ppo_head_0/advantages"/device:GPU:0*
T0

gmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/SumSumgmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Mulymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
š
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/ReshapeReshapegmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Sumimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Shape"/device:GPU:0*
T0*
Tshape0
ŗ
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Mul_1Mul?main_level/agent/main/online/network_1/ppo_head_0/clip_by_valueomain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Minimum_grad/Reshape_1"/device:GPU:0*
T0

imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Sum_1Sumimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Mul_1{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ö
mmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Reshape_1Reshapeimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Sum_1kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/ShapeShapeEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0

main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape_1ShapeMmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0_target"/device:GPU:0*
T0*
out_type0
č
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0
É
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/scalarConst^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape"/device:GPU:0*
valueB
 *   @*
dtype0
¶
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/MulMulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/scalar~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape"/device:GPU:0*
T0
æ
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/subSubEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAddMmain_level/agent/main/online/network_0/v_values_head_0/v_values_head_0_target^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape"/device:GPU:0*
T0
Ą
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/mul_1Mulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Mulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/sub"/device:GPU:0*
T0
ļ
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/SumSummain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/mul_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ö
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/ReshapeReshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Summain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape"/device:GPU:0*
T0*
Tshape0
ó
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Sum_1Summain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/mul_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ü
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape_1Reshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Sum_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
¹
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/NegNegmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape_1"/device:GPU:0*
T0
ė
qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/ShapeShapeGmain_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum"/device:GPU:0*
T0*
out_type0
«
smain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

smain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shape_2Shapekmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Reshape"/device:GPU:0*
T0*
out_type0
³
wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/zeros/ConstConst"/device:GPU:0*
valueB
 *    *
dtype0

qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/zerosFillsmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shape_2wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/zeros/Const"/device:GPU:0*
T0*

index_type0
¢
xmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/GreaterEqualGreaterEqualGmain_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum7main_level/agent/main/online/network_1/ppo_head_0/sub_1"/device:GPU:0*
T0

main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsqmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shapesmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shape_1"/device:GPU:0*
T0
ī
rmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/SelectSelectxmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/GreaterEqualkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Reshapeqmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/zeros"/device:GPU:0*
T0
£
omain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/SumSumrmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Selectmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

smain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/ReshapeReshapeomain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Sumqmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shape"/device:GPU:0*
T0*
Tshape0
š
tmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Select_1Selectxmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/GreaterEqualqmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/zeroskmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_3_grad/Reshape"/device:GPU:0*
T0
©
qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Sum_1Sumtmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Select_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

umain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Reshape_1Reshapeqmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Sum_1smain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ź
}main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape"/device:GPU:0*
T0*
data_formatNHWC
į
ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/ShapeShape5main_level/agent/main/online/network_1/ppo_head_0/Exp"/device:GPU:0*
T0*
out_type0
³
{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
”
{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_2Shapesmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Reshape"/device:GPU:0*
T0*
out_type0
»
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/zeros/ConstConst"/device:GPU:0*
valueB
 *    *
dtype0
©
ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/zerosFill{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_2main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/zeros/Const"/device:GPU:0*
T0*

index_type0

}main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/LessEqual	LessEqual5main_level/agent/main/online/network_1/ppo_head_0/Exp5main_level/agent/main/online/network_1/ppo_head_0/add"/device:GPU:0*
T0
³
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_1"/device:GPU:0*
T0

zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/SelectSelect}main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/LessEqualsmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Reshapeymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/zeros"/device:GPU:0*
T0
»
wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/SumSumzmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Selectmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
 
{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/ReshapeReshapewmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Sumymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape"/device:GPU:0*
T0*
Tshape0

|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Select_1Select}main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/LessEqualymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/zerossmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value_grad/Reshape"/device:GPU:0*
T0
Į
ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Sum_1Sum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Select_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
¦
}main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Reshape_1Reshapeymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Sum_1{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/ReshapeImain_level/agent/main/online/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul_1MatMulDmain_level/agent/main/online/network_0/v_values_head_0/strided_slicemain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
¾
+main_level/agent/main/online/gradients/AddNAddNkmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Reshape{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/clip_by_value/Minimum_grad/Reshape"/device:GPU:0*
T0*~
_classt
rploc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/mul_2_grad/Reshape*
N
č
emain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Exp_grad/mulMul+main_level/agent/main/online/gradients/AddN5main_level/agent/main/online/network_1/ppo_head_0/Exp"/device:GPU:0*
T0
Ó
vmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_0/add"/device:GPU:0*
T0*
out_type0
ų
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradvmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/ShapeJmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_2wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
ä
gmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/ShapeShapeJmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/Neg"/device:GPU:0*
T0*
out_type0
č
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Shape_1ShapeLmain_level/agent/main/online/network_1/ppo_head_0/Categorical_1/log_prob/Neg"/device:GPU:0*
T0*
out_type0
ü
wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Shapeimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Shape_1"/device:GPU:0*
T0

emain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/SumSumemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Exp_grad/mulwmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ź
imain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/ReshapeReshapeemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Sumgmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0

gmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Sum_1Sumemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Exp_grad/mulymain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ķ
emain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/NegNeggmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Sum_1"/device:GPU:0*
T0
ī
kmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Reshape_1Reshapeemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Negimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
¹
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/ShapeShape*main_level/agent/main/online/network_0/mul"/device:GPU:0*
T0*
out_type0
½
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape_1Shape,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/SumSummain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Sum_1Summain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape_1Reshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Sum_1^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/Neg_grad/NegNegimain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/sub_grad/Reshape"/device:GPU:0*
T0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ä
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape_1Shape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/MulMul^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
ą
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/SumSumZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Mullmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Mul_1Mul*main_level/agent/main/online/network_0/sub^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Sum_1Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Mul_1nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Reshape_1Reshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Sum_1^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Į
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/MulMul`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape_1.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/SumSum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Mulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/ReshapeReshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Sum^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
®
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/read`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Sum_1Sum^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Mul_1pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1Reshape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Sum_1`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
é
1main_level/agent/main/online/gradients/zeros_like	ZerosLikemain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1"/device:GPU:0*
T0
Ą
Źmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1"/device:GPU:0*“
messageØ„Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0

Émain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
Ą
Åmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimszmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/Neg_grad/NegÉmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0
ó
¾main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulÅmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsŹmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient"/device:GPU:0*
T0
Ł
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/sub_grad/NegNeg^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Reshape"/device:GPU:0*
T0
’
bmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1/y_grad/unstackUnpackbmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
÷
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/ShapeShapeHmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0*
out_type0

~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape_1ShapePmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like"/device:GPU:0*
T0*
out_type0
¼
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape_1"/device:GPU:0*
T0
¬
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/MulMul¾main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulPmain_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/ones_like"/device:GPU:0*
T0
Į
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/SumSumzmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Mulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
©
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/ReshapeReshapezmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Sum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
¦
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Mul_1MulHmain_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log¾main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"/device:GPU:0*
T0
Ē
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Sum_1Sum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Mul_1main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
°
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Reshape_1Reshape|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Sum_1~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

-main_level/agent/main/online/gradients/AddN_1AddN`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/ReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/sub_grad/Neg"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape*
N
ą
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log_grad/Reciprocal
Reciprocal8main_level/agent/main/online/network_1/ppo_head_0/policy^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Reshape"/device:GPU:0*
T0

xmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log_grad/mulMul~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Reshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log_grad/Reciprocal"/device:GPU:0*
T0

main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationbmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1/y_grad/unstack"/device:GPU:0*
T0
»
hmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mulMulxmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log_grad/mul8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0
¼
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0

hmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/SumSumhmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mulzmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
ė
hmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/subSubxmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/Categorical/logits/Log_grad/mulhmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum"/device:GPU:0*
T0
­
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1Mulhmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/sub8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0
į
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
„
{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGradBiasAddGradjmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
T0*
data_formatNHWC
·
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
³
main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
ņ
umain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMulMatMuljmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1Gmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
ģ
wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1MatMul?main_level/agent/main/online/network_1/ppo_head_0/strided_slicejmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
®
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ī
qmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_1/add"/device:GPU:0*
T0*
out_type0
Ü
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradqmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/ShapeEmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_2umain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
¹
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/ShapeShape*main_level/agent/main/online/network_1/mul"/device:GPU:0*
T0*
out_type0
½
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Shape_1Shape,main_level/agent/main/online/network_1/mul_1"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Shape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/SumSum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Sum_1Sum|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Reshape_1Reshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Sum_1^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ź
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ä
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Shape_1Shape3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Shape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/MulMul^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Reshape3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0
ą
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/SumSumZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Mullmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Mul_1Mul*main_level/agent/main/online/network_1/sub^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Reshape"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Sum_1Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Mul_1nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Reshape_1Reshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Sum_1^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Į
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Shape`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/MulMul`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Reshape_1.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/SumSum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Mulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/ReshapeReshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Sum^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
®
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/read`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/add_grad/Reshape_1"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Sum_1Sum^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Mul_1pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Reshape_1Reshape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Sum_1`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

tmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/online/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

vmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_0/observation/submain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Ł
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/sub_grad/NegNeg^main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_grad/Reshape"/device:GPU:0*
T0
’
bmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1/y_grad/unstackUnpackbmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

-main_level/agent/main/online/gradients/AddN_2AddN`main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/ReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/sub_grad/Neg"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Reshape*
N

main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationbmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1/y_grad/unstack"/device:GPU:0*
T0
į
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
·
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
³
main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
®
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul|main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation~main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ź
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

tmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/online/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

vmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_1/observation/submain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Õ
/main_level/agent/main/online/global_norm/L2LossL2Lossvmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
}{loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMul_1
į
1main_level/agent/main/online/global_norm/L2Loss_1L2Losszmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGrad
ń
1main_level/agent/main/online/global_norm/L2Loss_2L2Lossmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1
ł
1main_level/agent/main/online/global_norm/L2Loss_3L2Lossmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad
÷
1main_level/agent/main/online/global_norm/L2Loss_4L2Loss-main_level/agent/main/online/gradients/AddN_1"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape
ß
1main_level/agent/main/online/global_norm/L2Loss_5L2Lossymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
~loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul_1
č
1main_level/agent/main/online/global_norm/L2Loss_6L2Loss}main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGrad
×
1main_level/agent/main/online/global_norm/L2Loss_7L2Lossvmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
}{loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMul_1
į
1main_level/agent/main/online/global_norm/L2Loss_8L2Losszmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGrad
ń
1main_level/agent/main/online/global_norm/L2Loss_9L2Lossmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1
ś
2main_level/agent/main/online/global_norm/L2Loss_10L2Lossmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad
ų
2main_level/agent/main/online/global_norm/L2Loss_11L2Loss-main_level/agent/main/online/gradients/AddN_2"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/mul_1_grad/Reshape
Ū
2main_level/agent/main/online/global_norm/L2Loss_12L2Losswmain_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
~|loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1
å
2main_level/agent/main/online/global_norm/L2Loss_13L2Loss{main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGrad
Æ
.main_level/agent/main/online/global_norm/stackPack/main_level/agent/main/online/global_norm/L2Loss1main_level/agent/main/online/global_norm/L2Loss_11main_level/agent/main/online/global_norm/L2Loss_21main_level/agent/main/online/global_norm/L2Loss_31main_level/agent/main/online/global_norm/L2Loss_41main_level/agent/main/online/global_norm/L2Loss_51main_level/agent/main/online/global_norm/L2Loss_61main_level/agent/main/online/global_norm/L2Loss_71main_level/agent/main/online/global_norm/L2Loss_81main_level/agent/main/online/global_norm/L2Loss_92main_level/agent/main/online/global_norm/L2Loss_102main_level/agent/main/online/global_norm/L2Loss_112main_level/agent/main/online/global_norm/L2Loss_122main_level/agent/main/online/global_norm/L2Loss_13"/device:GPU:0*
T0*

axis *
N
k
.main_level/agent/main/online/global_norm/ConstConst"/device:GPU:0*
valueB: *
dtype0
Č
,main_level/agent/main/online/global_norm/SumSum.main_level/agent/main/online/global_norm/stack.main_level/agent/main/online/global_norm/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
l
0main_level/agent/main/online/global_norm/Const_1Const"/device:GPU:0*
valueB
 *   @*
dtype0
«
,main_level/agent/main/online/global_norm/mulMul,main_level/agent/main/online/global_norm/Sum0main_level/agent/main/online/global_norm/Const_1"/device:GPU:0*
T0

4main_level/agent/main/online/global_norm/global_normSqrt,main_level/agent/main/online/global_norm/mul"/device:GPU:0*
T0
¦
.main_level/agent/main/online/gradients_1/ShapeShapeEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_1/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_1/FillFill.main_level/agent/main/online/gradients_1/Shape2main_level/agent/main/online/gradients_1/grad_ys_0"/device:GPU:0*
T0*

index_type0
ģ
main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGradBiasAddGrad-main_level/agent/main/online/gradients_1/Fill"/device:GPU:0*
T0*
data_formatNHWC
»
ymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMulMatMul-main_level/agent/main/online/gradients_1/FillImain_level/agent/main/online/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
ø
{main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul_1MatMulDmain_level/agent/main/online/network_0/v_values_head_0/strided_slice-main_level/agent/main/online/gradients_1/Fill"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Õ
xmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_0/add"/device:GPU:0*
T0*
out_type0
ž
main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradxmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/ShapeJmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_2ymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
»
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/ShapeShape*main_level/agent/main/online/network_0/mul"/device:GPU:0*
T0*
out_type0
æ
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape_1Shape,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/SumSummain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/ReshapeReshape\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Sum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Sum_1Summain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Sum_1`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ę
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape_1Shape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/MulMul`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/SumSum\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Mulnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/ReshapeReshape\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Sum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Mul_1Mul*main_level/agent/main/online/network_0/sub`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Sum_1Sum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Mul_1pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Sum_1`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ć
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
ē
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shapebmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0

^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/MulMulbmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape_1.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/SumSum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Mulpmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Sum`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
²
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
ņ
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Mul_1rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ū
dmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Sum_1bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

dmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationdmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1/y_grad/unstack"/device:GPU:0*
T0
å
main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¼
main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
·
main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshapemain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¶
main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ī
|main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

vmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/online/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

xmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_0/observation/submain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
×
jmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/ShapeShape:main_level/agent/main/online/network_0/observation/truediv"/device:GPU:0*
T0*
out_type0
¤
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

zmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shapelmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0

hmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/SumSumvmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMulzmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/ReshapeReshapehmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Sumjmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0

jmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Sum_1Sumvmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMul|main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
hmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/NegNegjmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Sum_1"/device:GPU:0*
T0
÷
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Reshape_1Reshapehmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Neglmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
ß
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/ShapeShape>main_level/agent/main/online/network_0/observation/observation"/device:GPU:0*
T0*
out_type0
Ø
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shapepmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0
æ
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDivRealDivlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Reshape<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0

lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/SumSumpmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
’
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/ReshapeReshapelmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Sumnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ė
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/NegNeg>main_level/agent/main/online/network_0/observation/observation"/device:GPU:0*
T0
Į
rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_1RealDivlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Neg<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
Ē
rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_2RealDivrmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_1<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
ķ
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/mulMullmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Reshapermain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_2"/device:GPU:0*
T0

nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Sum_1Sumlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/mulmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Reshape_1Reshapenmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Sum_1pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
¦
.main_level/agent/main/online/gradients_2/ShapeShapeEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_2/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_2/FillFill.main_level/agent/main/online/gradients_2/Shape2main_level/agent/main/online/gradients_2/grad_ys_0"/device:GPU:0*
T0*

index_type0
¦
.main_level/agent/main/online/gradients_3/ShapeShapeEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_3/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_3/FillFill.main_level/agent/main/online/gradients_3/Shape2main_level/agent/main/online/gradients_3/grad_ys_0"/device:GPU:0*
T0*

index_type0
¦
.main_level/agent/main/online/gradients_4/ShapeShapeEmain_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_4/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_4/FillFill.main_level/agent/main/online/gradients_4/Shape2main_level/agent/main/online/gradients_4/grad_ys_0"/device:GPU:0*
T0*

index_type0

.main_level/agent/main/online/gradients_5/ShapeShape8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_5/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_5/FillFill.main_level/agent/main/online/gradients_5/Shape2main_level/agent/main/online/gradients_5/grad_ys_0"/device:GPU:0*
T0*

index_type0
ņ
jmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mulMul-main_level/agent/main/online/gradients_5/Fill8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0
¾
|main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0

jmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/SumSumjmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul|main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
¤
jmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/subSub-main_level/agent/main/online/gradients_5/Filljmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum"/device:GPU:0*
T0
±
lmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1Muljmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/sub8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0
©
}main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGradBiasAddGradlmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
T0*
data_formatNHWC
ö
wmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMulMatMullmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1Gmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
š
ymain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1MatMul?main_level/agent/main/online/network_1/ppo_head_0/strided_slicelmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Š
smain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_1/add"/device:GPU:0*
T0*
out_type0
ā
~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradsmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/ShapeEmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_2wmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
»
^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/ShapeShape*main_level/agent/main/online/network_1/mul"/device:GPU:0*
T0*
out_type0
æ
`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Shape_1Shape,main_level/agent/main/online/network_1/mul_1"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Shape`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/SumSum~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/ReshapeReshape\main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Sum^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Sum_1Sum~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Sum_1`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ę
`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Shape_1Shape3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Shape`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/MulMul`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Reshape3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/SumSum\main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Mulnmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/ReshapeReshape\main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Sum^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Mul_1Mul*main_level/agent/main/online/network_1/sub`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Reshape"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Sum_1Sum^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Mul_1pmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Sum_1`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ć
bmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0*
out_type0
ē
pmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Shapebmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0

^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/MulMulbmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Reshape_1.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/SumSum^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Mulpmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/ReshapeReshape^main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Sum`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
²
`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/add_grad/Reshape_1"/device:GPU:0*
T0
ņ
`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Sum_1Sum`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Mul_1rmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ū
dmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Reshape_1Reshape`main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Sum_1bmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

dmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1/y_grad/unstackUnpackdmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationdmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/mul_1/y_grad/unstack"/device:GPU:0*
T0
å
main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¼
main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
·
main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshapemain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¶
main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ī
|main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

vmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/online/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

xmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_1/observation/submain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
×
jmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/ShapeShape:main_level/agent/main/online/network_1/observation/truediv"/device:GPU:0*
T0*
out_type0
¤
lmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

zmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Shapelmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Shape_1"/device:GPU:0*
T0

hmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/SumSumvmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMulzmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
lmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/ReshapeReshapehmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Sumjmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0

jmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Sum_1Sumvmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMul|main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
hmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/NegNegjmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Sum_1"/device:GPU:0*
T0
÷
nmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Reshape_1Reshapehmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Neglmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
ß
nmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/ShapeShape>main_level/agent/main/online/network_0/observation/observation"/device:GPU:0*
T0*
out_type0
Ø
pmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Shapepmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Shape_1"/device:GPU:0*
T0
æ
pmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/RealDivRealDivlmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Reshape<main_level/agent/main/online/network_1/observation/truediv/y"/device:GPU:0*
T0

lmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/SumSumpmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/RealDiv~main_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
’
pmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/ReshapeReshapelmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Sumnmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ė
lmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/NegNeg>main_level/agent/main/online/network_0/observation/observation"/device:GPU:0*
T0
Į
rmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/RealDiv_1RealDivlmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Neg<main_level/agent/main/online/network_1/observation/truediv/y"/device:GPU:0*
T0
Ē
rmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/RealDiv_2RealDivrmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/RealDiv_1<main_level/agent/main/online/network_1/observation/truediv/y"/device:GPU:0*
T0
ķ
lmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/mulMullmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/sub_grad/Reshapermain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/RealDiv_2"/device:GPU:0*
T0

nmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Sum_1Sumlmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/mulmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

rmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Reshape_1Reshapenmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Sum_1pmain_level/agent/main/online/gradients_5/main_level/agent/main/online/network_1/observation/truediv_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

.main_level/agent/main/online/gradients_6/ShapeShape8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_6/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_6/FillFill.main_level/agent/main/online/gradients_6/Shape2main_level/agent/main/online/gradients_6/grad_ys_0"/device:GPU:0*
T0*

index_type0

.main_level/agent/main/online/gradients_7/ShapeShape8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_7/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_7/FillFill.main_level/agent/main/online/gradients_7/Shape2main_level/agent/main/online/gradients_7/grad_ys_0"/device:GPU:0*
T0*

index_type0

.main_level/agent/main/online/gradients_8/ShapeShape8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_8/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/online/gradients_8/FillFill.main_level/agent/main/online/gradients_8/Shape2main_level/agent/main/online/gradients_8/grad_ys_0"/device:GPU:0*
T0*

index_type0
}
4main_level/agent/main/online/output_gradient_weightsPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

6main_level/agent/main/online/output_gradient_weights_1Placeholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

2main_level/agent/main/online/gradients_9/grad_ys_0Identity4main_level/agent/main/online/output_gradient_weights"/device:GPU:0*
T0
ń
main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGradBiasAddGrad2main_level/agent/main/online/gradients_9/grad_ys_0"/device:GPU:0*
T0*
data_formatNHWC
Ą
ymain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMulMatMul2main_level/agent/main/online/gradients_9/grad_ys_0Imain_level/agent/main/online/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
½
{main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul_1MatMulDmain_level/agent/main/online/network_0/v_values_head_0/strided_slice2main_level/agent/main/online/gradients_9/grad_ys_0"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Õ
xmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_0/add"/device:GPU:0*
T0*
out_type0
ž
main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradxmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/ShapeJmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/online/network_0/v_values_head_0/strided_slice/stack_2ymain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/output/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
»
^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/ShapeShape*main_level/agent/main/online/network_0/mul"/device:GPU:0*
T0*
out_type0
æ
`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Shape_1Shape,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Shape`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/SumSummain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/ReshapeReshape\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Sum^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Sum_1Summain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Sum_1`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ę
`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Shape_1Shape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Shape`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/MulMul`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Reshape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
ę
\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/SumSum\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Mulnmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/ReshapeReshape\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Sum^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Mul_1Mul*main_level/agent/main/online/network_0/sub`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Reshape"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Sum_1Sum^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Mul_1pmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Sum_1`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ć
bmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
ē
pmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Shapebmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0

^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/MulMulbmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Reshape_1.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
ģ
^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/SumSum^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Mulpmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Sum`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
²
`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
ņ
`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Mul_1rmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ū
dmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Sum_1bmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ż
\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/sub_grad/NegNeg`main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_grad/Reshape"/device:GPU:0*
T0

dmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

-main_level/agent/main/online/gradients_9/AddNAddNbmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Reshape\main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/sub_grad/Neg"/device:GPU:0*
T0*u
_classk
igloc:@main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1_grad/Reshape*
N

main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationdmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/mul_1/y_grad/unstack"/device:GPU:0*
T0
å
main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¼
main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
·
main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshapemain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

~main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¶
main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul~main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ī
|main_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

vmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/online/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

xmain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_0/observation/submain_level/agent/main/online/gradients_9/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

3main_level/agent/main/online/gradients_10/grad_ys_0Identity6main_level/agent/main/online/output_gradient_weights_1"/device:GPU:0*
T0
ł
kmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mulMul3main_level/agent/main/online/gradients_10/grad_ys_08main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0
æ
}main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0

kmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/SumSumkmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul}main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
¬
kmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/subSub3main_level/agent/main/online/gradients_10/grad_ys_0kmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/Sum"/device:GPU:0*
T0
³
mmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1Mulkmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/sub8main_level/agent/main/online/network_1/ppo_head_0/policy"/device:GPU:0*
T0
«
~main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGradBiasAddGradmmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
T0*
data_formatNHWC
ų
xmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMulMatMulmmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1Gmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
ņ
zmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1MatMul?main_level/agent/main/online/network_1/ppo_head_0/strided_slicemmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Ń
tmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_1/add"/device:GPU:0*
T0*
out_type0
å
main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradtmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/ShapeEmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/online/network_1/ppo_head_0/strided_slice/stack_2xmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
¼
_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/ShapeShape*main_level/agent/main/online/network_1/mul"/device:GPU:0*
T0*
out_type0
Ą
amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Shape_1Shape,main_level/agent/main/online/network_1/mul_1"/device:GPU:0*
T0*
out_type0
ä
omain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Shapeamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Shape_1"/device:GPU:0*
T0

]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/SumSummain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradomain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ņ
amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/ReshapeReshape]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Sum_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Sum_1Summain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradqmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ų
cmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Reshape_1Reshape_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Sum_1amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ē
amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Shape_1Shape3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0*
out_type0
ä
omain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Shapeamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Shape_1"/device:GPU:0*
T0

]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/MulMulamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Reshape3main_level/agent/main/online/network_1/StopGradient"/device:GPU:0*
T0
é
]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/SumSum]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Mulomain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ņ
amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/ReshapeReshape]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Sum_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Mul_1Mul*main_level/agent/main/online/network_1/subamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Reshape"/device:GPU:0*
T0
ļ
_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Sum_1Sum_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Mul_1qmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ų
cmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Reshape_1Reshape_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Sum_1amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ä
cmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0*
out_type0
ź
qmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Shapecmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0

_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/MulMulcmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Reshape_1.main_level/agent/main/online/network_1/mul_1/y"/device:GPU:0*
T0
ļ
_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/SumSum_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Mulqmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ų
cmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/ReshapeReshape_main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Sumamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
“
amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/readcmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/add_grad/Reshape_1"/device:GPU:0*
T0
õ
amain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Sum_1Sumamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Mul_1smain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ž
emain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Reshape_1Reshapeamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Sum_1cmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
ß
]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/sub_grad/NegNegamain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_grad/Reshape"/device:GPU:0*
T0

emain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1/y_grad/unstackUnpackemain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

.main_level/agent/main/online/gradients_10/AddNAddNcmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Reshape]main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/sub_grad/Neg"/device:GPU:0*
T0*v
_classl
jhloc:@main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1_grad/Reshape*
N

main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationemain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/mul_1/y_grad/unstack"/device:GPU:0*
T0
ē
main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¾
main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
¹
main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/online/network_1/observation/Flatten/flatten/Reshapemain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¹
main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Š
}main_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

wmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/online/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

ymain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_1/observation/submain_level/agent/main/online/gradients_10/main_level/agent/main/online/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Ļ
6main_level/agent/main/online/beta1_power/initial_valueConst"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
valueB
 *fff?*
dtype0
ą
(main_level/agent/main/online/beta1_power
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container 
¹
/main_level/agent/main/online/beta1_power/AssignAssign(main_level/agent/main/online/beta1_power6main_level/agent/main/online/beta1_power/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
Ų
-main_level/agent/main/online/beta1_power/readIdentity(main_level/agent/main/online/beta1_power"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
Ļ
6main_level/agent/main/online/beta2_power/initial_valueConst"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
valueB
 *w¾?*
dtype0
ą
(main_level/agent/main/online/beta2_power
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container 
¹
/main_level/agent/main/online/beta2_power/AssignAssign(main_level/agent/main/online/beta2_power6main_level/agent/main/online/beta2_power/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
Ų
-main_level/agent/main/online/beta2_power/readIdentity(main_level/agent/main/online/beta2_power"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers

umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
dtype0

cmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
dtype0*
	container 
ē
jmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam/AssignAssigncmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adamumain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
validate_shape(
Ē
hmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam/readIdentitycmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel

wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
dtype0

emain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
dtype0*
	container 
ķ
lmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1/AssignAssignemain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
validate_shape(
Ė
jmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1/readIdentityemain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel

smain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
dtype0

amain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
dtype0*
	container 
ß
hmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam/AssignAssignamain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adamsmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
validate_shape(
Į
fmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam/readIdentityamain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias

umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
dtype0

cmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
dtype0*
	container 
å
jmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1/AssignAssigncmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
validate_shape(
Å
hmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1/readIdentitycmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias
¶
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"@   @   *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0
¤
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0
°
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zerosFillmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/shape_as_tensormain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
²
nmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 

umain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/AssignAssignnmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adammain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
č
smain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/readIdentitynmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
ø
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"@   @   *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0
¦
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0
¶
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zerosFillmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/shape_as_tensormain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel
“
pmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 

wmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/AssignAssignpmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
ģ
umain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/readIdentitypmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel

~main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
dtype0
Ŗ
lmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 

smain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam/AssignAssignlmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam~main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
validate_shape(
ā
qmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam/readIdentitylmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias
 
main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
dtype0
¬
nmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 

umain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1/AssignAssignnmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
validate_shape(
ę
smain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1/readIdentitynmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias

|main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Initializer/zerosConst"/device:GPU:0*
valueB
 *    *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0
¢
jmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container 

qmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/AssignAssignjmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam|main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
Ü
omain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/readIdentityjmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers

~main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB
 *    *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0
¤
lmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container 

smain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/AssignAssignlmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1~main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
ą
qmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/readIdentitylmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers

xmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
dtype0
¢
fmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam
VariableV2"/device:GPU:0*
shape
:@*
shared_name *W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
dtype0*
	container 
ó
mmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam/AssignAssignfmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adamxmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
validate_shape(
Š
kmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam/readIdentityfmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam"/device:GPU:0*
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel

zmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
dtype0
¤
hmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1
VariableV2"/device:GPU:0*
shape
:@*
shared_name *W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
dtype0*
	container 
ł
omain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1/AssignAssignhmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1zmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
validate_shape(
Ō
mmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1/readIdentityhmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1"/device:GPU:0*
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel

vmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam/Initializer/zerosConst"/device:GPU:0*
valueB*    *U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
dtype0

dmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam
VariableV2"/device:GPU:0*
shape:*
shared_name *U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
dtype0*
	container 
ė
kmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam/AssignAssigndmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adamvmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
validate_shape(
Ź
imain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam/readIdentitydmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias

xmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB*    *U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
dtype0

fmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1
VariableV2"/device:GPU:0*
shape:*
shared_name *U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
dtype0*
	container 
ń
mmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1/AssignAssignfmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1xmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
validate_shape(
Ī
kmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1/readIdentityfmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias

umain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
dtype0

cmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
dtype0*
	container 
ē
jmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam/AssignAssigncmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adamumain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
validate_shape(
Ē
hmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam/readIdentitycmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel

wmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
dtype0

emain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
dtype0*
	container 
ķ
lmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1/AssignAssignemain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1wmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
validate_shape(
Ė
jmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1/readIdentityemain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel

smain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
dtype0

amain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
dtype0*
	container 
ß
hmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam/AssignAssignamain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adamsmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
validate_shape(
Į
fmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam/readIdentityamain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias

umain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
dtype0

cmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
dtype0*
	container 
å
jmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1/AssignAssigncmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1umain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
validate_shape(
Å
hmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1/readIdentitycmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias
¶
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"@   @   *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0
¤
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0
°
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zerosFillmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/shape_as_tensormain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
²
nmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 

umain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/AssignAssignnmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adammain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
č
smain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/readIdentitynmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
ø
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"@   @   *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0
¦
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0
¶
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zerosFillmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/shape_as_tensormain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel
“
pmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 

wmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/AssignAssignpmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
ģ
umain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/readIdentitypmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel

~main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
dtype0
Ŗ
lmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 

smain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam/AssignAssignlmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam~main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
validate_shape(
ā
qmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam/readIdentitylmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias
 
main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
dtype0
¬
nmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 

umain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1/AssignAssignnmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
validate_shape(
ę
smain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1/readIdentitynmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias

|main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam/Initializer/zerosConst"/device:GPU:0*
valueB
 *    *[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
dtype0
¢
jmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
dtype0*
	container 

qmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam/AssignAssignjmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam|main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
validate_shape(
Ü
omain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam/readIdentityjmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers

~main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB
 *    *[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
dtype0
¤
lmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
dtype0*
	container 

smain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1/AssignAssignlmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1~main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
validate_shape(
ą
qmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1/readIdentitylmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers

vmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam/Initializer/zerosConst"/device:GPU:0*
valueB@*    *U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
dtype0

dmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam
VariableV2"/device:GPU:0*
shape
:@*
shared_name *U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
dtype0*
	container 
ė
kmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam/AssignAssigndmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adamvmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
validate_shape(
Ź
imain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam/readIdentitydmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel

xmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB@*    *U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
dtype0
 
fmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1
VariableV2"/device:GPU:0*
shape
:@*
shared_name *U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
dtype0*
	container 
ń
mmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1/AssignAssignfmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1xmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
validate_shape(
Ī
kmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1/readIdentityfmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel

tmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam/Initializer/zerosConst"/device:GPU:0*
valueB*    *S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
dtype0

bmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam
VariableV2"/device:GPU:0*
shape:*
shared_name *S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
dtype0*
	container 
ć
imain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam/AssignAssignbmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adamtmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
validate_shape(
Ä
gmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam/readIdentitybmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam"/device:GPU:0*
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias

vmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB*    *S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
dtype0

dmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1
VariableV2"/device:GPU:0*
shape:*
shared_name *S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
dtype0*
	container 
é
kmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1/AssignAssigndmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1vmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
validate_shape(
Č
imain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1/readIdentitydmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1"/device:GPU:0*
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias
k
/main_level/agent/main/online/Adam/learning_rateConst"/device:GPU:0*
valueB
 *RI9*
dtype0
c
'main_level/agent/main/online/Adam/beta1Const"/device:GPU:0*
valueB
 *fff?*
dtype0
c
'main_level/agent/main/online/Adam/beta2Const"/device:GPU:0*
valueB
 *w¾?*
dtype0
e
)main_level/agent/main/online/Adam/epsilonConst"/device:GPU:0*
valueB
 *¬Å'7*
dtype0
Ų
tmain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/kernel/ApplyAdam	ApplyAdamAmain_level/agent/main/online/network_0/observation/Dense_0/kernelcmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adamemain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/0_holder"/device:GPU:0*
use_locking( *
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_0/observation/Dense_0/kernel*
use_nesterov( 
Ī
rmain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/bias/ApplyAdam	ApplyAdam?main_level/agent/main/online/network_0/observation/Dense_0/biasamain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adamcmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/1_holder"/device:GPU:0*
use_locking( *
T0*R
_classH
FDloc:@main_level/agent/main/online/network_0/observation/Dense_0/bias*
use_nesterov( 

main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/ApplyAdam	ApplyAdamLmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernelnmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adampmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/2_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel*
use_nesterov( 

}main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/ApplyAdam	ApplyAdamJmain_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/biaslmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adamnmain_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/3_holder"/device:GPU:0*
use_locking( *
T0*]
_classS
QOloc:@main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias*
use_nesterov( 
ū
{main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam	ApplyAdamHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalersjmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adamlmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/4_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
use_nesterov( 
ē
wmain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/kernel/ApplyAdam	ApplyAdamDmain_level/agent/main/online/network_0/v_values_head_0/output/kernelfmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adamhmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/5_holder"/device:GPU:0*
use_locking( *
T0*W
_classM
KIloc:@main_level/agent/main/online/network_0/v_values_head_0/output/kernel*
use_nesterov( 
Ż
umain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/bias/ApplyAdam	ApplyAdamBmain_level/agent/main/online/network_0/v_values_head_0/output/biasdmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adamfmain_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/6_holder"/device:GPU:0*
use_locking( *
T0*U
_classK
IGloc:@main_level/agent/main/online/network_0/v_values_head_0/output/bias*
use_nesterov( 
Ų
tmain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/kernel/ApplyAdam	ApplyAdamAmain_level/agent/main/online/network_1/observation/Dense_0/kernelcmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adamemain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/7_holder"/device:GPU:0*
use_locking( *
T0*T
_classJ
HFloc:@main_level/agent/main/online/network_1/observation/Dense_0/kernel*
use_nesterov( 
Ī
rmain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/bias/ApplyAdam	ApplyAdam?main_level/agent/main/online/network_1/observation/Dense_0/biasamain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adamcmain_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/8_holder"/device:GPU:0*
use_locking( *
T0*R
_classH
FDloc:@main_level/agent/main/online/network_1/observation/Dense_0/bias*
use_nesterov( 

main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/ApplyAdam	ApplyAdamLmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernelnmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adampmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/9_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel*
use_nesterov( 

}main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/ApplyAdam	ApplyAdamJmain_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/biaslmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adamnmain_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/10_holder"/device:GPU:0*
use_locking( *
T0*]
_classS
QOloc:@main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias*
use_nesterov( 
ü
{main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/ApplyAdam	ApplyAdamHmain_level/agent/main/online/network_1/gradients_from_head_0-0_rescalersjmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adamlmain_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/11_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers*
use_nesterov( 
Ž
umain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/ApplyAdam	ApplyAdamBmain_level/agent/main/online/network_1/ppo_head_0/policy_fc/kerneldmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adamfmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/12_holder"/device:GPU:0*
use_locking( *
T0*U
_classK
IGloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel*
use_nesterov( 
Ō
smain_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/ApplyAdam	ApplyAdam@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/biasbmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adamdmain_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/13_holder"/device:GPU:0*
use_locking( *
T0*S
_classI
GEloc:@main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias*
use_nesterov( 
³
%main_level/agent/main/online/Adam/mulMul-main_level/agent/main/online/beta1_power/read'main_level/agent/main/online/Adam/beta1|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam~^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/ApplyAdam^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/ApplyAdams^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/bias/ApplyAdamu^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/kernel/ApplyAdamv^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/bias/ApplyAdamx^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/kernel/ApplyAdam|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/ApplyAdam~^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/ApplyAdam^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/ApplyAdams^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/bias/ApplyAdamu^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/kernel/ApplyAdamt^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/ApplyAdamv^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/ApplyAdam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
”
(main_level/agent/main/online/Adam/AssignAssign(main_level/agent/main/online/beta1_power%main_level/agent/main/online/Adam/mul"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
µ
'main_level/agent/main/online/Adam/mul_1Mul-main_level/agent/main/online/beta2_power/read'main_level/agent/main/online/Adam/beta2|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam~^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/ApplyAdam^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/ApplyAdams^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/bias/ApplyAdamu^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/kernel/ApplyAdamv^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/bias/ApplyAdamx^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/kernel/ApplyAdam|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/ApplyAdam~^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/ApplyAdam^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/ApplyAdams^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/bias/ApplyAdamu^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/kernel/ApplyAdamt^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/ApplyAdamv^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/ApplyAdam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
„
*main_level/agent/main/online/Adam/Assign_1Assign(main_level/agent/main/online/beta2_power'main_level/agent/main/online/Adam/mul_1"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
Ń
(main_level/agent/main/online/Adam/updateNoOp)^main_level/agent/main/online/Adam/Assign+^main_level/agent/main/online/Adam/Assign_1|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam~^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/ApplyAdam^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/ApplyAdams^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/bias/ApplyAdamu^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/Dense_0/kernel/ApplyAdamv^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/bias/ApplyAdamx^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/v_values_head_0/output/kernel/ApplyAdam|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/ApplyAdam~^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/ApplyAdam^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/ApplyAdams^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/bias/ApplyAdamu^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/observation/Dense_0/kernel/ApplyAdamt^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/ApplyAdamv^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/ApplyAdam"/device:GPU:0
Č
'main_level/agent/main/online/Adam/valueConst)^main_level/agent/main/online/Adam/update"/device:GPU:0*;
_class1
/-loc:@main_level/agent/main/online/global_step*
value	B	 R*
dtype0	
é
!main_level/agent/main/online/Adam	AssignAdd(main_level/agent/main/online/global_step'main_level/agent/main/online/Adam/value"/device:GPU:0*
use_locking( *
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
e
,main_level/agent/main/online/AssignAdd/valueConst"/device:GPU:0*
value	B	 R*
dtype0	
ó
&main_level/agent/main/online/AssignAdd	AssignAdd(main_level/agent/main/online/global_step,main_level/agent/main/online/AssignAdd/value"/device:GPU:0*
use_locking( *
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
#
!main_level/agent/main/online/initNoOp0^main_level/agent/main/online/beta1_power/Assign0^main_level/agent/main/online/beta2_power/Assign0^main_level/agent/main/online/global_step/Assignr^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Assigni^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam/Assignm^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1/Assignl^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam/Assignn^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1/Assignn^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam/Assignp^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1/Assignr^main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Assigni^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam/Assignm^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1/Assignj^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam/Assignl^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1/Assignl^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam/Assignn^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1/AssignP^main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/AssignR^main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/AssignT^main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/AssignG^main_level/agent/main/online/network_0/observation/Dense_0/bias/AssignI^main_level/agent/main/online/network_0/observation/Dense_0/kernel/AssignJ^main_level/agent/main/online/network_0/v_values_head_0/output/bias/AssignL^main_level/agent/main/online/network_0/v_values_head_0/output/kernel/AssignP^main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/AssignR^main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/AssignT^main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/AssignG^main_level/agent/main/online/network_1/observation/Dense_0/bias/AssignI^main_level/agent/main/online/network_1/observation/Dense_0/kernel/AssignH^main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/AssignJ^main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Assign"/device:GPU:0
Ū
#main_level/agent/main/online/init_1NoOp-^main_level/agent/main/online/Variable/Assign7^main_level/agent/main/online/network_0/Variable/Assign7^main_level/agent/main/online/network_1/Variable/Assign"/device:GPU:0

'main_level/agent/main/online/group_depsNoOp"^main_level/agent/main/online/init$^main_level/agent/main/online/init_1"/device:GPU:0
l
3main_level/agent/main/target/Variable/initial_valueConst"/device:GPU:0*
value	B
 Z *
dtype0


%main_level/agent/main/target/Variable
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0
*
	container 

,main_level/agent/main/target/Variable/AssignAssign%main_level/agent/main/target/Variable3main_level/agent/main/target/Variable/initial_value"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/target/Variable*
validate_shape(
Æ
*main_level/agent/main/target/Variable/readIdentity%main_level/agent/main/target/Variable"/device:GPU:0*
T0
*8
_class.
,*loc:@main_level/agent/main/target/Variable
b
(main_level/agent/main/target/PlaceholderPlaceholder"/device:GPU:0*
shape:*
dtype0

ł
#main_level/agent/main/target/AssignAssign%main_level/agent/main/target/Variable(main_level/agent/main/target/Placeholder"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/target/Variable*
validate_shape(

>main_level/agent/main/target/network_0/observation/observationPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
x
<main_level/agent/main/target/network_0/observation/truediv/yConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ū
:main_level/agent/main/target/network_0/observation/truedivRealDiv>main_level/agent/main/target/network_0/observation/observation<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
t
8main_level/agent/main/target/network_0/observation/sub/yConst"/device:GPU:0*
valueB
 *    *
dtype0
Ė
6main_level/agent/main/target/network_0/observation/subSub:main_level/agent/main/target/network_0/observation/truediv8main_level/agent/main/target/network_0/observation/sub/y"/device:GPU:0*
T0
ķ
bmain_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/shapeConst*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
valueB"   @   *
dtype0
ć
`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/minConst*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
valueB
 *0¾*
dtype0
ć
`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/maxConst*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
valueB
 *0>*
dtype0
ä
jmain_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformbmain_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
dtype0*
seed2 

`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/subSub`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/max`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel

`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/mulMuljmain_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniform`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/sub*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel

\main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniformAdd`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/mul`main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel
ś
Amain_level/agent/main/target/network_0/observation/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
dtype0*
	container 

Hmain_level/agent/main/target/network_0/observation/Dense_0/kernel/AssignAssignAmain_level/agent/main/target/network_0/observation/Dense_0/kernel\main_level/agent/main/target/network_0/observation/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
validate_shape(

Fmain_level/agent/main/target/network_0/observation/Dense_0/kernel/readIdentityAmain_level/agent/main/target/network_0/observation/Dense_0/kernel"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel
Ö
Qmain_level/agent/main/target/network_0/observation/Dense_0/bias/Initializer/zerosConst*R
_classH
FDloc:@main_level/agent/main/target/network_0/observation/Dense_0/bias*
valueB@*    *
dtype0
ņ
?main_level/agent/main/target/network_0/observation/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/target/network_0/observation/Dense_0/bias*
dtype0*
	container 
ł
Fmain_level/agent/main/target/network_0/observation/Dense_0/bias/AssignAssign?main_level/agent/main/target/network_0/observation/Dense_0/biasQmain_level/agent/main/target/network_0/observation/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/target/network_0/observation/Dense_0/bias*
validate_shape(
ż
Dmain_level/agent/main/target/network_0/observation/Dense_0/bias/readIdentity?main_level/agent/main/target/network_0/observation/Dense_0/bias"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/target/network_0/observation/Dense_0/bias

Amain_level/agent/main/target/network_0/observation/Dense_0/MatMulMatMul6main_level/agent/main/target/network_0/observation/subFmain_level/agent/main/target/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Bmain_level/agent/main/target/network_0/observation/Dense_0/BiasAddBiasAddAmain_level/agent/main/target/network_0/observation/Dense_0/MatMulDmain_level/agent/main/target/network_0/observation/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
¾
Zmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activationTanhBmain_level/agent/main/target/network_0/observation/Dense_0/BiasAdd"/device:GPU:0*
T0
Õ
Hmain_level/agent/main/target/network_0/observation/Flatten/flatten/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0

Vmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Xmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Xmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
æ
Pmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_sliceStridedSliceHmain_level/agent/main/target/network_0/observation/Flatten/flatten/ShapeVmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stackXmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_1Xmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

Rmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shape/1Const"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
«
Pmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shapePackPmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_sliceRmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shape/1"/device:GPU:0*
T0*

axis *
N
©
Jmain_level/agent/main/target/network_0/observation/Flatten/flatten/ReshapeReshapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activationPmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shape"/device:GPU:0*
T0*
Tshape0

mmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shapeConst*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
valueB"@   @   *
dtype0
ł
kmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/minConst*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]¾*
dtype0
ł
kmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxConst*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]>*
dtype0

umain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformmmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0*
seed2 
¶
kmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/subSubkmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxkmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel
Ą
kmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulMulumain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformkmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/sub*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel
²
gmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniformAddkmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulkmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel

Lmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 
¶
Smain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/AssignAssignLmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernelgmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
¤
Qmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/readIdentityLmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel
ģ
\main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias/Initializer/zerosConst*]
_classS
QOloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias*
valueB@*    *
dtype0

Jmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 
„
Qmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias/AssignAssignJmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias\main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias*
validate_shape(

Omain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias/readIdentityJmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias
³
Lmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMulMatMulJmain_level/agent/main/target/network_0/observation/Flatten/flatten/ReshapeQmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 
¦
Mmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAddBiasAddLmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMulOmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
Ō
emain_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationTanhMmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAdd"/device:GPU:0*
T0
}
=main_level/agent/main/target/network_0/Variable/initial_valueConst"/device:GPU:0*
valueB*  ?*
dtype0

/main_level/agent/main/target/network_0/Variable
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
	container 
µ
6main_level/agent/main/target/network_0/Variable/AssignAssign/main_level/agent/main/target/network_0/Variable=main_level/agent/main/target/network_0/Variable/initial_value"/device:GPU:0*
use_locking(*
T0*B
_class8
64loc:@main_level/agent/main/target/network_0/Variable*
validate_shape(
Ķ
4main_level/agent/main/target/network_0/Variable/readIdentity/main_level/agent/main/target/network_0/Variable"/device:GPU:0*
T0*B
_class8
64loc:@main_level/agent/main/target/network_0/Variable
C
Const_2Const"/device:GPU:0*
valueB
 *  ?*
dtype0

Vmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/initial_valueConst"/device:GPU:0*
valueB
 *  ?*
dtype0
£
Hmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0*
	container 

Omain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/AssignAssignHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalersVmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers*
validate_shape(

Mmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/readIdentityHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers

Jmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers_1Placeholder"/device:GPU:0*
shape:*
dtype0
ė
-main_level/agent/main/target/network_0/AssignAssignHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalersJmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers_1"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
h
,main_level/agent/main/target/network_0/sub/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ę
*main_level/agent/main/target/network_0/subSub,main_level/agent/main/target/network_0/sub/xMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/read"/device:GPU:0*
T0
Õ
9main_level/agent/main/target/network_0/StopGradient/inputPackemain_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N

3main_level/agent/main/target/network_0/StopGradientStopGradient9main_level/agent/main/target/network_0/StopGradient/input"/device:GPU:0*
T0
Ŗ
*main_level/agent/main/target/network_0/mulMul*main_level/agent/main/target/network_0/sub3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
Ź
.main_level/agent/main/target/network_0/mul_1/yPackemain_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N
Ź
,main_level/agent/main/target/network_0/mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/read.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
£
*main_level/agent/main/target/network_0/addAdd*main_level/agent/main/target/network_0/mul,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0

Jmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
ń
Dmain_level/agent/main/target/network_0/v_values_head_0/strided_sliceStridedSlice*main_level/agent/main/target/network_0/addJmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
ä
Vmain_level/agent/main/target/network_0/v_values_head_0/output/kernel/Initializer/ConstConst*W
_classM
KIloc:@main_level/agent/main/target/network_0/v_values_head_0/output/kernel*
valueB@""«f¾,>ŽwĶ½V,>é¾CQ>cü¢>”·=i¾Š~N½ 8>²V¹»r/=d=(>óY”=yu½Ō¤½»>Ć<ņ}d>Ūö½'ē=(į ¼Ö
Ó=·<½`>Ö¼h:=-_Ö½}4<¼D·m¾QŚŪ=«f{½^·ö=}¾ö)>Ė½²¼¢Ł¾ŌE<Æā=²t¾d¹>;·*K½¢«=sA>ź>Ž½×O;1÷4¾­+>“ø½yż½r“=yź"½ÜUŌ=ó½øaĆ=»¤@>øĶN¾Ö×;²)¾*
dtype0

Dmain_level/agent/main/target/network_0/v_values_head_0/output/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *W
_classM
KIloc:@main_level/agent/main/target/network_0/v_values_head_0/output/kernel*
dtype0*
	container 

Kmain_level/agent/main/target/network_0/v_values_head_0/output/kernel/AssignAssignDmain_level/agent/main/target/network_0/v_values_head_0/output/kernelVmain_level/agent/main/target/network_0/v_values_head_0/output/kernel/Initializer/Const"/device:GPU:0*
use_locking(*
T0*W
_classM
KIloc:@main_level/agent/main/target/network_0/v_values_head_0/output/kernel*
validate_shape(

Imain_level/agent/main/target/network_0/v_values_head_0/output/kernel/readIdentityDmain_level/agent/main/target/network_0/v_values_head_0/output/kernel"/device:GPU:0*
T0*W
_classM
KIloc:@main_level/agent/main/target/network_0/v_values_head_0/output/kernel
Ü
Tmain_level/agent/main/target/network_0/v_values_head_0/output/bias/Initializer/zerosConst*U
_classK
IGloc:@main_level/agent/main/target/network_0/v_values_head_0/output/bias*
valueB*    *
dtype0
ų
Bmain_level/agent/main/target/network_0/v_values_head_0/output/bias
VariableV2"/device:GPU:0*
shape:*
shared_name *U
_classK
IGloc:@main_level/agent/main/target/network_0/v_values_head_0/output/bias*
dtype0*
	container 

Imain_level/agent/main/target/network_0/v_values_head_0/output/bias/AssignAssignBmain_level/agent/main/target/network_0/v_values_head_0/output/biasTmain_level/agent/main/target/network_0/v_values_head_0/output/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_0/v_values_head_0/output/bias*
validate_shape(

Gmain_level/agent/main/target/network_0/v_values_head_0/output/bias/readIdentityBmain_level/agent/main/target/network_0/v_values_head_0/output/bias"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_0/v_values_head_0/output/bias

Dmain_level/agent/main/target/network_0/v_values_head_0/output/MatMulMatMulDmain_level/agent/main/target/network_0/v_values_head_0/strided_sliceImain_level/agent/main/target/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Emain_level/agent/main/target/network_0/v_values_head_0/output/BiasAddBiasAddDmain_level/agent/main/target/network_0/v_values_head_0/output/MatMulGmain_level/agent/main/target/network_0/v_values_head_0/output/bias/read"/device:GPU:0*
T0*
data_formatNHWC

Mmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0_targetPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
”
Xmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0_importance_weightPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1/stackConst"/device:GPU:0*
valueB: *
dtype0

Nmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Nmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1/stack_2Const"/device:GPU:0*
valueB:*
dtype0

Fmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1StridedSlice4main_level/agent/main/target/network_0/Variable/readLmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1/stackNmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1/stack_1Nmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
ū
:main_level/agent/main/target/network_0/v_values_head_0/mulMulFmain_level/agent/main/target/network_0/v_values_head_0/strided_slice_1Xmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0_importance_weight"/device:GPU:0*
T0

Xmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifferenceSquaredDifferenceEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAddMmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0_target"/device:GPU:0*
T0

cmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/weightsConst"/device:GPU:0*
valueB
 *  ?*
dtype0
”
imain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/weights/shapeConst"/device:GPU:0*
valueB *
dtype0
”
hmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/weights/rankConst"/device:GPU:0*
value	B : *
dtype0
ó
hmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/values/shapeShapeXmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference"/device:GPU:0*
T0*
out_type0
 
gmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/values/rankConst"/device:GPU:0*
value	B :*
dtype0

wmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/static_scalar_check_successNoOp"/device:GPU:0

Mmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Cast/xConstx^main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/assert_broadcastable/static_scalar_check_success"/device:GPU:0*
valueB
 *  ?*
dtype0

Jmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/MulMulXmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifferenceMmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Cast/x"/device:GPU:0*
T0

Lmain_level/agent/main/target/network_0/v_values_head_0/Sum/reduction_indicesConst"/device:GPU:0*
valueB:*
dtype0

:main_level/agent/main/target/network_0/v_values_head_0/SumSumJmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/MulLmain_level/agent/main/target/network_0/v_values_head_0/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ó
<main_level/agent/main/target/network_0/v_values_head_0/mul_1Mul:main_level/agent/main/target/network_0/v_values_head_0/mul:main_level/agent/main/target/network_0/v_values_head_0/Sum"/device:GPU:0*
T0

<main_level/agent/main/target/network_0/v_values_head_0/ConstConst"/device:GPU:0*
valueB"       *
dtype0
ō
;main_level/agent/main/target/network_0/v_values_head_0/MeanMean<main_level/agent/main/target/network_0/v_values_head_0/mul_1<main_level/agent/main/target/network_0/v_values_head_0/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
x
<main_level/agent/main/target/network_1/observation/truediv/yConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ū
:main_level/agent/main/target/network_1/observation/truedivRealDiv>main_level/agent/main/target/network_0/observation/observation<main_level/agent/main/target/network_1/observation/truediv/y"/device:GPU:0*
T0
t
8main_level/agent/main/target/network_1/observation/sub/yConst"/device:GPU:0*
valueB
 *    *
dtype0
Ė
6main_level/agent/main/target/network_1/observation/subSub:main_level/agent/main/target/network_1/observation/truediv8main_level/agent/main/target/network_1/observation/sub/y"/device:GPU:0*
T0
ķ
bmain_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/shapeConst*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
valueB"   @   *
dtype0
ć
`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/minConst*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
valueB
 *0¾*
dtype0
ć
`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/maxConst*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
valueB
 *0>*
dtype0
ä
jmain_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformbmain_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
dtype0*
seed2 

`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/subSub`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/max`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel

`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/mulMuljmain_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/RandomUniform`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/sub*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel

\main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniformAdd`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/mul`main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel
ś
Amain_level/agent/main/target/network_1/observation/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
dtype0*
	container 

Hmain_level/agent/main/target/network_1/observation/Dense_0/kernel/AssignAssignAmain_level/agent/main/target/network_1/observation/Dense_0/kernel\main_level/agent/main/target/network_1/observation/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
validate_shape(

Fmain_level/agent/main/target/network_1/observation/Dense_0/kernel/readIdentityAmain_level/agent/main/target/network_1/observation/Dense_0/kernel"/device:GPU:0*
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel
Ö
Qmain_level/agent/main/target/network_1/observation/Dense_0/bias/Initializer/zerosConst*R
_classH
FDloc:@main_level/agent/main/target/network_1/observation/Dense_0/bias*
valueB@*    *
dtype0
ņ
?main_level/agent/main/target/network_1/observation/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *R
_classH
FDloc:@main_level/agent/main/target/network_1/observation/Dense_0/bias*
dtype0*
	container 
ł
Fmain_level/agent/main/target/network_1/observation/Dense_0/bias/AssignAssign?main_level/agent/main/target/network_1/observation/Dense_0/biasQmain_level/agent/main/target/network_1/observation/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*R
_classH
FDloc:@main_level/agent/main/target/network_1/observation/Dense_0/bias*
validate_shape(
ż
Dmain_level/agent/main/target/network_1/observation/Dense_0/bias/readIdentity?main_level/agent/main/target/network_1/observation/Dense_0/bias"/device:GPU:0*
T0*R
_classH
FDloc:@main_level/agent/main/target/network_1/observation/Dense_0/bias

Amain_level/agent/main/target/network_1/observation/Dense_0/MatMulMatMul6main_level/agent/main/target/network_1/observation/subFmain_level/agent/main/target/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Bmain_level/agent/main/target/network_1/observation/Dense_0/BiasAddBiasAddAmain_level/agent/main/target/network_1/observation/Dense_0/MatMulDmain_level/agent/main/target/network_1/observation/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
¾
Zmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activationTanhBmain_level/agent/main/target/network_1/observation/Dense_0/BiasAdd"/device:GPU:0*
T0
Õ
Hmain_level/agent/main/target/network_1/observation/Flatten/flatten/ShapeShapeZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0

Vmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Xmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Xmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
æ
Pmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_sliceStridedSliceHmain_level/agent/main/target/network_1/observation/Flatten/flatten/ShapeVmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_slice/stackXmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_slice/stack_1Xmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

Rmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape/shape/1Const"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
«
Pmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape/shapePackPmain_level/agent/main/target/network_1/observation/Flatten/flatten/strided_sliceRmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape/shape/1"/device:GPU:0*
T0*

axis *
N
©
Jmain_level/agent/main/target/network_1/observation/Flatten/flatten/ReshapeReshapeZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activationPmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape/shape"/device:GPU:0*
T0*
Tshape0

mmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shapeConst*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
valueB"@   @   *
dtype0
ł
kmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/minConst*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]¾*
dtype0
ł
kmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxConst*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
valueB
 *×³]>*
dtype0

umain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformmmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/shape*

seed *
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0*
seed2 
¶
kmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/subSubkmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/maxkmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel
Ą
kmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulMulumain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/RandomUniformkmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/sub*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel
²
gmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniformAddkmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/mulkmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform/min*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel

Lmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel
VariableV2"/device:GPU:0*
shape
:@@*
shared_name *_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
dtype0*
	container 
¶
Smain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/AssignAssignLmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernelgmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
¤
Qmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/readIdentityLmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel
ģ
\main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias/Initializer/zerosConst*]
_classS
QOloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias*
valueB@*    *
dtype0

Jmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias
VariableV2"/device:GPU:0*
shape:@*
shared_name *]
_classS
QOloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias*
dtype0*
	container 
„
Qmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias/AssignAssignJmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias\main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*]
_classS
QOloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias*
validate_shape(

Omain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias/readIdentityJmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias"/device:GPU:0*
T0*]
_classS
QOloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias
³
Lmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMulMatMulJmain_level/agent/main/target/network_1/observation/Flatten/flatten/ReshapeQmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 
¦
Mmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAddBiasAddLmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMulOmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias/read"/device:GPU:0*
T0*
data_formatNHWC
Ō
emain_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationTanhMmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAdd"/device:GPU:0*
T0
}
=main_level/agent/main/target/network_1/Variable/initial_valueConst"/device:GPU:0*
valueB*  ?*
dtype0

/main_level/agent/main/target/network_1/Variable
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
	container 
µ
6main_level/agent/main/target/network_1/Variable/AssignAssign/main_level/agent/main/target/network_1/Variable=main_level/agent/main/target/network_1/Variable/initial_value"/device:GPU:0*
use_locking(*
T0*B
_class8
64loc:@main_level/agent/main/target/network_1/Variable*
validate_shape(
Ķ
4main_level/agent/main/target/network_1/Variable/readIdentity/main_level/agent/main/target/network_1/Variable"/device:GPU:0*
T0*B
_class8
64loc:@main_level/agent/main/target/network_1/Variable
C
Const_3Const"/device:GPU:0*
valueB
 *  ?*
dtype0

Vmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/initial_valueConst"/device:GPU:0*
valueB
 *  ?*
dtype0
£
Hmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0*
	container 

Omain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/AssignAssignHmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalersVmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers*
validate_shape(

Mmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/readIdentityHmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers

Hmain_level/agent/main/target/network_1/gradients_from_head_1-0_rescalersPlaceholder"/device:GPU:0*
shape:*
dtype0
é
-main_level/agent/main/target/network_1/AssignAssignHmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalersHmain_level/agent/main/target/network_1/gradients_from_head_1-0_rescalers"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers*
validate_shape(
h
,main_level/agent/main/target/network_1/sub/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ę
*main_level/agent/main/target/network_1/subSub,main_level/agent/main/target/network_1/sub/xMmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/read"/device:GPU:0*
T0
Õ
9main_level/agent/main/target/network_1/StopGradient/inputPackemain_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N

3main_level/agent/main/target/network_1/StopGradientStopGradient9main_level/agent/main/target/network_1/StopGradient/input"/device:GPU:0*
T0
Ŗ
*main_level/agent/main/target/network_1/mulMul*main_level/agent/main/target/network_1/sub3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0
Ź
.main_level/agent/main/target/network_1/mul_1/yPackemain_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*

axis *
N
Ź
,main_level/agent/main/target/network_1/mul_1MulMmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/read.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0
£
*main_level/agent/main/target/network_1/addAdd*main_level/agent/main/target/network_1/mul,main_level/agent/main/target/network_1/mul_1"/device:GPU:0*
T0

Emain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0

Gmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0

Gmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
Ż
?main_level/agent/main/target/network_1/ppo_head_0/strided_sliceStridedSlice*main_level/agent/main/target/network_1/addEmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
~
9main_level/agent/main/target/network_1/ppo_head_0/actionsPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

Amain_level/agent/main/target/network_1/ppo_head_0/old_policy_meanPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

@main_level/agent/main/target/network_1/ppo_head_0/old_policy_stdPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
ļ
cmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/shapeConst*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
valueB"@      *
dtype0
å
amain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/minConst*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
valueB
 *²_¾*
dtype0
å
amain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/maxConst*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
valueB
 *²_>*
dtype0
ē
kmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniformcmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/shape*

seed *
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
dtype0*
seed2 

amain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/subSubamain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/maxamain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/min*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel

amain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/mulMulkmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/RandomUniformamain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/sub*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel

]main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniformAddamain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/mulamain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform/min*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel
ü
Bmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel
VariableV2"/device:GPU:0*
shape
:@*
shared_name *U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
dtype0*
	container 

Imain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/AssignAssignBmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel]main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
validate_shape(

Gmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/readIdentityBmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel"/device:GPU:0*
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel
Ų
Rmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias/Initializer/zerosConst*S
_classI
GEloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias*
valueB*    *
dtype0
ō
@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias
VariableV2"/device:GPU:0*
shape:*
shared_name *S
_classI
GEloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias*
dtype0*
	container 
ż
Gmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias/AssignAssign@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/biasRmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*S
_classI
GEloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias*
validate_shape(

Emain_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias/readIdentity@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias"/device:GPU:0*
T0*S
_classI
GEloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias

Bmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMulMatMul?main_level/agent/main/target/network_1/ppo_head_0/strided_sliceGmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 

Cmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAddBiasAddBmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMulEmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias/read"/device:GPU:0*
T0*
data_formatNHWC
 
8main_level/agent/main/target/network_1/ppo_head_0/policySoftmaxCmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAdd"/device:GPU:0*
T0
”
Hmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/LogLog8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0

Hmain_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_rankConst"/device:GPU:0*
value	B :*
dtype0
Å
Jmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits_shapeShapeHmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0*
out_type0

Hmain_level/agent/main/target/network_1/ppo_head_0/Categorical/event_sizeConst"/device:GPU:0*
value	B :*
dtype0

]main_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
„
_main_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

_main_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
Ż
Wmain_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_sliceStridedSliceJmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits_shape]main_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_main_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_1_main_level/agent/main/target/network_1/ppo_head_0/Categorical/batch_shape/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 
¬
Jmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits/LogLogAmain_level/agent/main/target/network_1/ppo_head_0/old_policy_mean"/device:GPU:0*
T0

Jmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_rankConst"/device:GPU:0*
value	B :*
dtype0
É
Lmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits_shapeShapeJmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits/Log"/device:GPU:0*
T0*
out_type0

Jmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/event_sizeConst"/device:GPU:0*
value	B :*
dtype0

_main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
§
amain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

amain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
ē
Ymain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_sliceStridedSliceLmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits_shape_main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stackamain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_1amain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/batch_shape/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

Zmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stackConst"/device:GPU:0*
valueB"        *
dtype0
 
\main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_1Const"/device:GPU:0*
valueB"        *
dtype0
 
\main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_2Const"/device:GPU:0*
valueB"      *
dtype0
Ą
Tmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_sliceStridedSlice9main_level/agent/main/target/network_1/ppo_head_0/actionsZmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack\main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_1\main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask*
new_axis_mask*
end_mask 
Ż
Vmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like/ShapeShapeTmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice"/device:GPU:0*
T0*
out_type0

Vmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like/ConstConst"/device:GPU:0*
valueB
 *  ?*
dtype0
²
Pmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_likeFillVmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like/ShapeVmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like/Const"/device:GPU:0*
T0*

index_type0

Jmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mulMulHmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/LogPmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like"/device:GPU:0*
T0
É
Lmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ShapeShapeJmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul"/device:GPU:0*
T0*
out_type0

\main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stackConst"/device:GPU:0*
valueB: *
dtype0
¤
^main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

^main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_2Const"/device:GPU:0*
valueB:*
dtype0
Ū
Vmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1StridedSliceLmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/Shape\main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack^main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_1^main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

Qmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones/ConstConst"/device:GPU:0*
value	B :*
dtype0
Ø
Kmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/onesFillVmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/strided_slice_1Qmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones/Const"/device:GPU:0*
T0*

index_type0
ó
Lmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_1Mul9main_level/agent/main/target/network_1/ppo_head_0/actionsKmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones"/device:GPU:0*
T0
ļ
pmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeLmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_1"/device:GPU:0*
T0*
out_type0
÷
main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsJmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mulLmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_1"/device:GPU:0*
T0*
Tlabels0
ś
Jmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/NegNegmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"/device:GPU:0*
T0
 
\main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stackConst"/device:GPU:0*
valueB"        *
dtype0
¢
^main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_1Const"/device:GPU:0*
valueB"        *
dtype0
¢
^main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_2Const"/device:GPU:0*
valueB"      *
dtype0
Č
Vmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_sliceStridedSlice9main_level/agent/main/target/network_1/ppo_head_0/actions\main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack^main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_1^main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask*
new_axis_mask*
end_mask 
į
Xmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/ShapeShapeVmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice"/device:GPU:0*
T0*
out_type0

Xmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/ConstConst"/device:GPU:0*
valueB
 *  ?*
dtype0
ø
Rmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones_likeFillXmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/ShapeXmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones_like/Const"/device:GPU:0*
T0*

index_type0

Lmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/mulMulJmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits/LogRmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones_like"/device:GPU:0*
T0
Ķ
Nmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ShapeShapeLmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/mul"/device:GPU:0*
T0*
out_type0

^main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stackConst"/device:GPU:0*
valueB: *
dtype0
¦
`main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_1Const"/device:GPU:0*
valueB:
’’’’’’’’’*
dtype0

`main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_2Const"/device:GPU:0*
valueB:*
dtype0
å
Xmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1StridedSliceNmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/Shape^main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack`main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_1`main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 

Smain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones/ConstConst"/device:GPU:0*
value	B :*
dtype0
®
Mmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/onesFillXmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/strided_slice_1Smain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones/Const"/device:GPU:0*
T0*

index_type0
÷
Nmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/mul_1Mul9main_level/agent/main/target/network_1/ppo_head_0/actionsMmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/ones"/device:GPU:0*
T0
ó
rmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeNmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/mul_1"/device:GPU:0*
T0*
out_type0
ż
main_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsLmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/mulNmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/mul_1"/device:GPU:0*
T0*
Tlabels0
ž
Lmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/NegNegmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"/device:GPU:0*
T0
Ą
Pmain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/LogSoftmax
LogSoftmaxHmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0
ō
Imain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/mulMulPmain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/LogSoftmax8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0

[main_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
­
Imain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/SumSumImain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/mul[main_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
³
Imain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/NegNegImain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/Sum"/device:GPU:0*
T0
t
7main_level/agent/main/target/network_1/ppo_head_0/ConstConst"/device:GPU:0*
valueB: *
dtype0
÷
6main_level/agent/main/target/network_1/ppo_head_0/MeanMeanImain_level/agent/main/target/network_1/ppo_head_0/Categorical/entropy/Neg7main_level/agent/main/target/network_1/ppo_head_0/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ł
gmain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmax
LogSoftmaxJmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits/Log"/device:GPU:0*
T0
Ł
imain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmax_1
LogSoftmaxHmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0
Ó
`main_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/subSubgmain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmaximain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/LogSoftmax_1"/device:GPU:0*
T0
Ó
dmain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/SoftmaxSoftmaxJmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/logits/Log"/device:GPU:0*
T0
Ē
`main_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/mulMuldmain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Softmax`main_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/sub"/device:GPU:0*
T0
“
rmain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
ņ
`main_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/SumSum`main_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/mulrmain_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
v
9main_level/agent/main/target/network_1/ppo_head_0/Const_1Const"/device:GPU:0*
valueB: *
dtype0

8main_level/agent/main/target/network_1/ppo_head_0/Mean_1Mean`main_level/agent/main/target/network_1/ppo_head_0/KullbackLeibler/kl_categorical_categorical/Sum9main_level/agent/main/target/network_1/ppo_head_0/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

<main_level/agent/main/target/network_1/ppo_head_0/advantagesPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
ī
5main_level/agent/main/target/network_1/ppo_head_0/subSubJmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/NegLmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/Neg"/device:GPU:0*
T0

5main_level/agent/main/target/network_1/ppo_head_0/ExpExp5main_level/agent/main/target/network_1/ppo_head_0/sub"/device:GPU:0*
T0
u
=main_level/agent/main/target/network_1/ppo_head_0/PlaceholderPlaceholder"/device:GPU:0*
shape: *
dtype0
s
7main_level/agent/main/target/network_1/ppo_head_0/mul/xConst"/device:GPU:0*
valueB
 *ĶĢL>*
dtype0
Ģ
5main_level/agent/main/target/network_1/ppo_head_0/mulMul7main_level/agent/main/target/network_1/ppo_head_0/mul/x=main_level/agent/main/target/network_1/ppo_head_0/Placeholder"/device:GPU:0*
T0
s
7main_level/agent/main/target/network_1/ppo_head_0/add/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ä
5main_level/agent/main/target/network_1/ppo_head_0/addAdd7main_level/agent/main/target/network_1/ppo_head_0/add/x5main_level/agent/main/target/network_1/ppo_head_0/mul"/device:GPU:0*
T0
u
9main_level/agent/main/target/network_1/ppo_head_0/mul_1/xConst"/device:GPU:0*
valueB
 *ĶĢL>*
dtype0
Š
7main_level/agent/main/target/network_1/ppo_head_0/mul_1Mul9main_level/agent/main/target/network_1/ppo_head_0/mul_1/x=main_level/agent/main/target/network_1/ppo_head_0/Placeholder"/device:GPU:0*
T0
u
9main_level/agent/main/target/network_1/ppo_head_0/sub_1/xConst"/device:GPU:0*
valueB
 *  ?*
dtype0
Ź
7main_level/agent/main/target/network_1/ppo_head_0/sub_1Sub9main_level/agent/main/target/network_1/ppo_head_0/sub_1/x7main_level/agent/main/target/network_1/ppo_head_0/mul_1"/device:GPU:0*
T0
Ų
Gmain_level/agent/main/target/network_1/ppo_head_0/clip_by_value/MinimumMinimum5main_level/agent/main/target/network_1/ppo_head_0/Exp5main_level/agent/main/target/network_1/ppo_head_0/add"/device:GPU:0*
T0
ä
?main_level/agent/main/target/network_1/ppo_head_0/clip_by_valueMaximumGmain_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum7main_level/agent/main/target/network_1/ppo_head_0/sub_1"/device:GPU:0*
T0
Ė
7main_level/agent/main/target/network_1/ppo_head_0/mul_2Mul5main_level/agent/main/target/network_1/ppo_head_0/Exp<main_level/agent/main/target/network_1/ppo_head_0/advantages"/device:GPU:0*
T0
Õ
7main_level/agent/main/target/network_1/ppo_head_0/mul_3Mul?main_level/agent/main/target/network_1/ppo_head_0/clip_by_value<main_level/agent/main/target/network_1/ppo_head_0/advantages"/device:GPU:0*
T0
Ī
9main_level/agent/main/target/network_1/ppo_head_0/MinimumMinimum7main_level/agent/main/target/network_1/ppo_head_0/mul_27main_level/agent/main/target/network_1/ppo_head_0/mul_3"/device:GPU:0*
T0
v
9main_level/agent/main/target/network_1/ppo_head_0/Const_2Const"/device:GPU:0*
valueB: *
dtype0
ė
8main_level/agent/main/target/network_1/ppo_head_0/Mean_2Mean9main_level/agent/main/target/network_1/ppo_head_0/Minimum9main_level/agent/main/target/network_1/ppo_head_0/Const_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

5main_level/agent/main/target/network_1/ppo_head_0/NegNeg8main_level/agent/main/target/network_1/ppo_head_0/Mean_2"/device:GPU:0*
T0

Nmain_level/agent/main/target/network_1/ppo_head_0/ppo_head_0_importance_weightPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0
Ń
(main_level/agent/main/target/Rank/packedPack;main_level/agent/main/target/network_0/v_values_head_0/Mean5main_level/agent/main/target/network_1/ppo_head_0/Neg"/device:GPU:0*
T0*

axis *
N
Z
!main_level/agent/main/target/RankConst"/device:GPU:0*
value	B :*
dtype0
a
(main_level/agent/main/target/range/startConst"/device:GPU:0*
value	B : *
dtype0
a
(main_level/agent/main/target/range/deltaConst"/device:GPU:0*
value	B :*
dtype0
½
"main_level/agent/main/target/rangeRange(main_level/agent/main/target/range/start!main_level/agent/main/target/Rank(main_level/agent/main/target/range/delta"/device:GPU:0*

Tidx0
Ļ
&main_level/agent/main/target/Sum/inputPack;main_level/agent/main/target/network_0/v_values_head_0/Mean5main_level/agent/main/target/network_1/ppo_head_0/Neg"/device:GPU:0*
T0*

axis *
N
Ø
 main_level/agent/main/target/SumSum&main_level/agent/main/target/Sum/input"main_level/agent/main/target/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
e
%main_level/agent/main/target/0_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
a
%main_level/agent/main/target/1_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
e
%main_level/agent/main/target/2_holderPlaceholder"/device:GPU:0*
shape
:@@*
dtype0
a
%main_level/agent/main/target/3_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
]
%main_level/agent/main/target/4_holderPlaceholder"/device:GPU:0*
shape: *
dtype0
e
%main_level/agent/main/target/5_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
a
%main_level/agent/main/target/6_holderPlaceholder"/device:GPU:0*
shape:*
dtype0
e
%main_level/agent/main/target/7_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
a
%main_level/agent/main/target/8_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
e
%main_level/agent/main/target/9_holderPlaceholder"/device:GPU:0*
shape
:@@*
dtype0
b
&main_level/agent/main/target/10_holderPlaceholder"/device:GPU:0*
shape:@*
dtype0
^
&main_level/agent/main/target/11_holderPlaceholder"/device:GPU:0*
shape: *
dtype0
f
&main_level/agent/main/target/12_holderPlaceholder"/device:GPU:0*
shape
:@*
dtype0
b
&main_level/agent/main/target/13_holderPlaceholder"/device:GPU:0*
shape:*
dtype0
°
%main_level/agent/main/target/Assign_1AssignAmain_level/agent/main/target/network_0/observation/Dense_0/kernel%main_level/agent/main/target/0_holder"/device:GPU:0*
use_locking( *
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_0/observation/Dense_0/kernel*
validate_shape(
¬
%main_level/agent/main/target/Assign_2Assign?main_level/agent/main/target/network_0/observation/Dense_0/bias%main_level/agent/main/target/1_holder"/device:GPU:0*
use_locking( *
T0*R
_classH
FDloc:@main_level/agent/main/target/network_0/observation/Dense_0/bias*
validate_shape(
Ę
%main_level/agent/main/target/Assign_3AssignLmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel%main_level/agent/main/target/2_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
Ā
%main_level/agent/main/target/Assign_4AssignJmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias%main_level/agent/main/target/3_holder"/device:GPU:0*
use_locking( *
T0*]
_classS
QOloc:@main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias*
validate_shape(
¾
%main_level/agent/main/target/Assign_5AssignHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers%main_level/agent/main/target/4_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
¶
%main_level/agent/main/target/Assign_6AssignDmain_level/agent/main/target/network_0/v_values_head_0/output/kernel%main_level/agent/main/target/5_holder"/device:GPU:0*
use_locking( *
T0*W
_classM
KIloc:@main_level/agent/main/target/network_0/v_values_head_0/output/kernel*
validate_shape(
²
%main_level/agent/main/target/Assign_7AssignBmain_level/agent/main/target/network_0/v_values_head_0/output/bias%main_level/agent/main/target/6_holder"/device:GPU:0*
use_locking( *
T0*U
_classK
IGloc:@main_level/agent/main/target/network_0/v_values_head_0/output/bias*
validate_shape(
°
%main_level/agent/main/target/Assign_8AssignAmain_level/agent/main/target/network_1/observation/Dense_0/kernel%main_level/agent/main/target/7_holder"/device:GPU:0*
use_locking( *
T0*T
_classJ
HFloc:@main_level/agent/main/target/network_1/observation/Dense_0/kernel*
validate_shape(
¬
%main_level/agent/main/target/Assign_9Assign?main_level/agent/main/target/network_1/observation/Dense_0/bias%main_level/agent/main/target/8_holder"/device:GPU:0*
use_locking( *
T0*R
_classH
FDloc:@main_level/agent/main/target/network_1/observation/Dense_0/bias*
validate_shape(
Ē
&main_level/agent/main/target/Assign_10AssignLmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel%main_level/agent/main/target/9_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel*
validate_shape(
Ä
&main_level/agent/main/target/Assign_11AssignJmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias&main_level/agent/main/target/10_holder"/device:GPU:0*
use_locking( *
T0*]
_classS
QOloc:@main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias*
validate_shape(
Ą
&main_level/agent/main/target/Assign_12AssignHmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers&main_level/agent/main/target/11_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers*
validate_shape(
“
&main_level/agent/main/target/Assign_13AssignBmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel&main_level/agent/main/target/12_holder"/device:GPU:0*
use_locking( *
T0*U
_classK
IGloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel*
validate_shape(
°
&main_level/agent/main/target/Assign_14Assign@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias&main_level/agent/main/target/13_holder"/device:GPU:0*
use_locking( *
T0*S
_classI
GEloc:@main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias*
validate_shape(
d
,main_level/agent/main/target/gradients/ShapeConst"/device:GPU:0*
valueB *
dtype0
l
0main_level/agent/main/target/gradients/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
½
+main_level/agent/main/target/gradients/FillFill,main_level/agent/main/target/gradients/Shape0main_level/agent/main/target/gradients/grad_ys_0"/device:GPU:0*
T0*

index_type0

Zmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0

Tmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/ReshapeReshape+main_level/agent/main/target/gradients/FillZmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0

Rmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/ConstConst"/device:GPU:0*
valueB:*
dtype0
­
Qmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/TileTileTmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/ReshapeRmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Const"/device:GPU:0*

Tmultiples0*
T0
ę
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum/input_grad/unstackUnpackQmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Tile"/device:GPU:0*
T0*	
num*

axis 
¹
umain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Reshape/shapeConst"/device:GPU:0*
valueB"      *
dtype0
ó
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum/input_grad/unstackumain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0
Ü
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/ShapeShape<main_level/agent/main/target/network_0/v_values_head_0/mul_1"/device:GPU:0*
T0*
out_type0
ž
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/TileTileomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Reshapemmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Shape"/device:GPU:0*

Tmultiples0*
T0
Ž
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Shape_1Shape<main_level/agent/main/target/network_0/v_values_head_0/mul_1"/device:GPU:0*
T0*
out_type0
§
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Shape_2Const"/device:GPU:0*
valueB *
dtype0
Ŗ
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0

lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/ProdProdomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Shape_1mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
¬
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0

nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Prod_1Prodomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Shape_2omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ŗ
qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Maximum/yConst"/device:GPU:0*
value	B :*
dtype0
õ
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/MaximumMaximumnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Prod_1qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Maximum/y"/device:GPU:0*
T0
ó
pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/floordivFloorDivlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Prodomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Maximum"/device:GPU:0*
T0

lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/CastCastpmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/floordiv"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
ī
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/truedivRealDivlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Tilelmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/Cast"/device:GPU:0*
T0
ā
emain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Neg_grad/NegNeg\main_level/agent/main/target/gradients/main_level/agent/main/target/Sum/input_grad/unstack:1"/device:GPU:0*
T0
Ū
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/ShapeShape:main_level/agent/main/target/network_0/v_values_head_0/mul"/device:GPU:0*
T0*
out_type0
Ż
pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Shape_1Shape:main_level/agent/main/target/network_0/v_values_head_0/Sum"/device:GPU:0*
T0*
out_type0

~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Shapepmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
ø
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/MulMulomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/truediv:main_level/agent/main/target/network_0/v_values_head_0/Sum"/device:GPU:0*
T0

lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/SumSumlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Mul~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
’
pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/ReshapeReshapelmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Sumnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
ŗ
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Mul_1Mul:main_level/agent/main/target/network_0/v_values_head_0/mulomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Mean_grad/truediv"/device:GPU:0*
T0

nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Sum_1Sumnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Mul_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

rmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Reshape_1Reshapenmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Sum_1pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Æ
rmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0
ų
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/ReshapeReshapeemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Neg_grad/Negrmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0
Ö
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/ShapeShape9main_level/agent/main/target/network_1/ppo_head_0/Minimum"/device:GPU:0*
T0*
out_type0
õ
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/TileTilelmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Reshapejmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Shape"/device:GPU:0*

Tmultiples0*
T0
Ų
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Shape_1Shape9main_level/agent/main/target/network_1/ppo_head_0/Minimum"/device:GPU:0*
T0*
out_type0
¤
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Shape_2Const"/device:GPU:0*
valueB *
dtype0
§
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0

imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/ProdProdlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Shape_1jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
©
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0

kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Prod_1Prodlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Shape_2lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
§
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Maximum/yConst"/device:GPU:0*
value	B :*
dtype0
ģ
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/MaximumMaximumkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Prod_1nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Maximum/y"/device:GPU:0*
T0
ź
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/floordivFloorDivimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Prodlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Maximum"/device:GPU:0*
T0

imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/CastCastmmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/floordiv"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
å
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/truedivRealDivimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Tileimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/Cast"/device:GPU:0*
T0
é
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/ShapeShapeJmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul"/device:GPU:0*
T0*
out_type0
„
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/SizeConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0
Å
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/addAddLmain_level/agent/main/target/network_0/v_values_head_0/Sum/reduction_indiceskmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Size"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape
č
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/modFloorModjmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/addkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Size"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape
¬
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape_1Const"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
valueB:*
dtype0
¬
rmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/range/startConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
value	B : *
dtype0
¬
rmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/range/deltaConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0
ę
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/rangeRangermain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/range/startkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Sizermain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/range/delta"/device:GPU:0*

Tidx0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape
«
qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Fill/valueConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0

kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/FillFillnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape_1qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Fill/value"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*

index_type0
Ü
tmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/DynamicStitchDynamicStitchlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/rangejmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/modlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shapekmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Fill"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
N
Ŗ
pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Maximum/yConst"/device:GPU:0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape*
value	B :*
dtype0
ś
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/MaximumMaximumtmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/DynamicStitchpmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Maximum/y"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape
ņ
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/floordivFloorDivlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shapenmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Maximum"/device:GPU:0*
T0*
_classu
sqloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Shape

nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/ReshapeReshapermain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/mul_1_grad/Reshape_1tmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0
ž
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/TileTilenmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Reshapeomain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
Õ
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/ShapeShape7main_level/agent/main/target/network_1/ppo_head_0/mul_2"/device:GPU:0*
T0*
out_type0
×
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shape_1Shape7main_level/agent/main/target/network_1/ppo_head_0/mul_3"/device:GPU:0*
T0*
out_type0

mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shape_2Shapelmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/truediv"/device:GPU:0*
T0*
out_type0
­
qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/zeros/ConstConst"/device:GPU:0*
valueB
 *    *
dtype0
’
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/zerosFillmmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shape_2qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/zeros/Const"/device:GPU:0*
T0*

index_type0

omain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/LessEqual	LessEqual7main_level/agent/main/target/network_1/ppo_head_0/mul_27main_level/agent/main/target/network_1/ppo_head_0/mul_3"/device:GPU:0*
T0

{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgskmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shapemmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shape_1"/device:GPU:0*
T0
Ś
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/SelectSelectomain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/LessEquallmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/truedivkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/zeros"/device:GPU:0*
T0

imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/SumSumlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Select{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ö
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/ReshapeReshapeimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Sumkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ü
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Select_1Selectomain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/LessEqualkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/zeroslmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Mean_2_grad/truediv"/device:GPU:0*
T0

kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Sum_1Sumnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Select_1}main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ü
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Reshape_1Reshapekmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Sum_1mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/ShapeShapeXmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference"/device:GPU:0*
T0*
out_type0
¶
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
¼
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape_1"/device:GPU:0*
T0
Õ
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/MulMulkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/TileMmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Cast/x"/device:GPU:0*
T0
Į
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/SumSumzmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Mulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
©
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/ReshapeReshapezmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Sum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
ā
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Mul_1MulXmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifferencekmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/Sum_grad/Tile"/device:GPU:0*
T0
Ē
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Sum_1Sum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Mul_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
°
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape_1Reshape|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Sum_1~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ń
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/ShapeShape5main_level/agent/main/target/network_1/ppo_head_0/Exp"/device:GPU:0*
T0*
out_type0
Ś
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Shape_1Shape<main_level/agent/main/target/network_1/ppo_head_0/advantages"/device:GPU:0*
T0*
out_type0

ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Shapekmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Shape_1"/device:GPU:0*
T0
³
gmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/MulMulmmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Reshape<main_level/agent/main/target/network_1/ppo_head_0/advantages"/device:GPU:0*
T0

gmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/SumSumgmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Mulymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
š
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/ReshapeReshapegmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Sumimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
®
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Mul_1Mul5main_level/agent/main/target/network_1/ppo_head_0/Expmmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Reshape"/device:GPU:0*
T0

imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Sum_1Sumimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Mul_1{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ö
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Reshape_1Reshapeimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Sum_1kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ū
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/ShapeShape?main_level/agent/main/target/network_1/ppo_head_0/clip_by_value"/device:GPU:0*
T0*
out_type0
Ś
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Shape_1Shape<main_level/agent/main/target/network_1/ppo_head_0/advantages"/device:GPU:0*
T0*
out_type0

ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Shapekmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Shape_1"/device:GPU:0*
T0
µ
gmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/MulMulomain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Reshape_1<main_level/agent/main/target/network_1/ppo_head_0/advantages"/device:GPU:0*
T0

gmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/SumSumgmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Mulymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
š
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/ReshapeReshapegmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Sumimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Shape"/device:GPU:0*
T0*
Tshape0
ŗ
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Mul_1Mul?main_level/agent/main/target/network_1/ppo_head_0/clip_by_valueomain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Minimum_grad/Reshape_1"/device:GPU:0*
T0

imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Sum_1Sumimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Mul_1{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ö
mmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Reshape_1Reshapeimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Sum_1kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/ShapeShapeEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0

main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape_1ShapeMmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0_target"/device:GPU:0*
T0*
out_type0
č
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0
É
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/scalarConst^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape"/device:GPU:0*
valueB
 *   @*
dtype0
¶
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/MulMulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/scalar~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape"/device:GPU:0*
T0
æ
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/subSubEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAddMmain_level/agent/main/target/network_0/v_values_head_0/v_values_head_0_target^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/Mul_grad/Reshape"/device:GPU:0*
T0
Ą
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/mul_1Mulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Mulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/sub"/device:GPU:0*
T0
ļ
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/SumSummain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/mul_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ö
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/ReshapeReshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Summain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape"/device:GPU:0*
T0*
Tshape0
ó
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Sum_1Summain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/mul_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ü
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape_1Reshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Sum_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
¹
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/NegNegmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape_1"/device:GPU:0*
T0
ė
qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/ShapeShapeGmain_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum"/device:GPU:0*
T0*
out_type0
«
smain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

smain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shape_2Shapekmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Reshape"/device:GPU:0*
T0*
out_type0
³
wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/zeros/ConstConst"/device:GPU:0*
valueB
 *    *
dtype0

qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/zerosFillsmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shape_2wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/zeros/Const"/device:GPU:0*
T0*

index_type0
¢
xmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/GreaterEqualGreaterEqualGmain_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum7main_level/agent/main/target/network_1/ppo_head_0/sub_1"/device:GPU:0*
T0

main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsqmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shapesmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shape_1"/device:GPU:0*
T0
ī
rmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/SelectSelectxmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/GreaterEqualkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Reshapeqmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/zeros"/device:GPU:0*
T0
£
omain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/SumSumrmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Selectmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

smain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/ReshapeReshapeomain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Sumqmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shape"/device:GPU:0*
T0*
Tshape0
š
tmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Select_1Selectxmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/GreaterEqualqmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/zeroskmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_3_grad/Reshape"/device:GPU:0*
T0
©
qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Sum_1Sumtmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Select_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

umain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Reshape_1Reshapeqmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Sum_1smain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ź
}main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape"/device:GPU:0*
T0*
data_formatNHWC
į
ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/ShapeShape5main_level/agent/main/target/network_1/ppo_head_0/Exp"/device:GPU:0*
T0*
out_type0
³
{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
”
{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_2Shapesmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Reshape"/device:GPU:0*
T0*
out_type0
»
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/zeros/ConstConst"/device:GPU:0*
valueB
 *    *
dtype0
©
ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/zerosFill{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_2main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/zeros/Const"/device:GPU:0*
T0*

index_type0

}main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/LessEqual	LessEqual5main_level/agent/main/target/network_1/ppo_head_0/Exp5main_level/agent/main/target/network_1/ppo_head_0/add"/device:GPU:0*
T0
³
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_1"/device:GPU:0*
T0

zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/SelectSelect}main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/LessEqualsmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Reshapeymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/zeros"/device:GPU:0*
T0
»
wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/SumSumzmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Selectmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
 
{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/ReshapeReshapewmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Sumymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape"/device:GPU:0*
T0*
Tshape0

|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Select_1Select}main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/LessEqualymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/zerossmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value_grad/Reshape"/device:GPU:0*
T0
Į
ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Sum_1Sum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Select_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
¦
}main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Reshape_1Reshapeymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Sum_1{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/ReshapeImain_level/agent/main/target/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul_1MatMulDmain_level/agent/main/target/network_0/v_values_head_0/strided_slicemain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/v_values_head_0/SquaredDifference_grad/Reshape"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
¾
+main_level/agent/main/target/gradients/AddNAddNkmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Reshape{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/clip_by_value/Minimum_grad/Reshape"/device:GPU:0*
T0*~
_classt
rploc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/mul_2_grad/Reshape*
N
č
emain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Exp_grad/mulMul+main_level/agent/main/target/gradients/AddN5main_level/agent/main/target/network_1/ppo_head_0/Exp"/device:GPU:0*
T0
Ó
vmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_0/add"/device:GPU:0*
T0*
out_type0
ų
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradvmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/ShapeJmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_2wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
ä
gmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/ShapeShapeJmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/Neg"/device:GPU:0*
T0*
out_type0
č
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Shape_1ShapeLmain_level/agent/main/target/network_1/ppo_head_0/Categorical_1/log_prob/Neg"/device:GPU:0*
T0*
out_type0
ü
wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Shapeimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Shape_1"/device:GPU:0*
T0

emain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/SumSumemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Exp_grad/mulwmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ź
imain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/ReshapeReshapeemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Sumgmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0

gmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Sum_1Sumemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Exp_grad/mulymain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ķ
emain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/NegNeggmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Sum_1"/device:GPU:0*
T0
ī
kmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Reshape_1Reshapeemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Negimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
¹
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/ShapeShape*main_level/agent/main/target/network_0/mul"/device:GPU:0*
T0*
out_type0
½
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape_1Shape,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/SumSummain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Sum_1Summain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape_1Reshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Sum_1^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/Neg_grad/NegNegimain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/sub_grad/Reshape"/device:GPU:0*
T0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ä
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape_1Shape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/MulMul^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
ą
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/SumSumZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Mullmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Mul_1Mul*main_level/agent/main/target/network_0/sub^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Sum_1Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Mul_1nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Reshape_1Reshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Sum_1^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Į
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/MulMul`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape_1.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/SumSum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Mulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/ReshapeReshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Sum^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
®
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/read`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Sum_1Sum^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Mul_1pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1Reshape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Sum_1`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
é
1main_level/agent/main/target/gradients/zeros_like	ZerosLikemain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1"/device:GPU:0*
T0
Ą
Źmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1"/device:GPU:0*“
messageØ„Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0

Émain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0
Ą
Åmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimszmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/Neg_grad/NegÉmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0
ó
¾main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulÅmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsŹmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient"/device:GPU:0*
T0
Ł
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/sub_grad/NegNeg^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Reshape"/device:GPU:0*
T0
’
bmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1/y_grad/unstackUnpackbmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
÷
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/ShapeShapeHmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log"/device:GPU:0*
T0*
out_type0

~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape_1ShapePmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like"/device:GPU:0*
T0*
out_type0
¼
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape_1"/device:GPU:0*
T0
¬
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/MulMul¾main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulPmain_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/ones_like"/device:GPU:0*
T0
Į
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/SumSumzmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Mulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
©
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/ReshapeReshapezmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Sum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
¦
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Mul_1MulHmain_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log¾main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"/device:GPU:0*
T0
Ē
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Sum_1Sum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Mul_1main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
°
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Reshape_1Reshape|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Sum_1~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

-main_level/agent/main/target/gradients/AddN_1AddN`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/ReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/sub_grad/Neg"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape*
N
ą
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log_grad/Reciprocal
Reciprocal8main_level/agent/main/target/network_1/ppo_head_0/policy^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Reshape"/device:GPU:0*
T0

xmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log_grad/mulMul~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/log_prob/mul_grad/Reshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log_grad/Reciprocal"/device:GPU:0*
T0

main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationbmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1/y_grad/unstack"/device:GPU:0*
T0
»
hmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mulMulxmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log_grad/mul8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0
¼
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0

hmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/SumSumhmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mulzmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
ė
hmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/subSubxmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/Categorical/logits/Log_grad/mulhmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum"/device:GPU:0*
T0
­
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1Mulhmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/sub8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0
į
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
„
{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGradBiasAddGradjmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
T0*
data_formatNHWC
·
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
³
main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
ņ
umain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMulMatMuljmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1Gmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
ģ
wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1MatMul?main_level/agent/main/target/network_1/ppo_head_0/strided_slicejmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
®
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ī
qmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_1/add"/device:GPU:0*
T0*
out_type0
Ü
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradqmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/ShapeEmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_2umain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
¹
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/ShapeShape*main_level/agent/main/target/network_1/mul"/device:GPU:0*
T0*
out_type0
½
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Shape_1Shape,main_level/agent/main/target/network_1/mul_1"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Shape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/SumSum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Sum_1Sum|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Reshape_1Reshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Sum_1^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ź
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ä
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Shape_1Shape3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0*
out_type0
Ū
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Shape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Shape_1"/device:GPU:0*
T0

Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/MulMul^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Reshape3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0
ą
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/SumSumZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Mullmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
É
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Mul_1Mul*main_level/agent/main/target/network_1/sub^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Reshape"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Sum_1Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Mul_1nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Reshape_1Reshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Sum_1^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Į
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Shape`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/MulMul`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Reshape_1.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/SumSum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Mulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/ReshapeReshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Sum^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
®
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/read`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/add_grad/Reshape_1"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Sum_1Sum^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Mul_1pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Reshape_1Reshape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Sum_1`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

tmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/target/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

vmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_0/observation/submain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Ł
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/sub_grad/NegNeg^main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_grad/Reshape"/device:GPU:0*
T0
’
bmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1/y_grad/unstackUnpackbmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

-main_level/agent/main/target/gradients/AddN_2AddN`main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/ReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/sub_grad/Neg"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Reshape*
N

main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationbmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1/y_grad/unstack"/device:GPU:0*
T0
į
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
·
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
³
main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
®
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul|main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation~main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ź
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

tmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/target/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

vmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_1/observation/submain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Õ
/main_level/agent/main/target/global_norm/L2LossL2Lossvmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
}{loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMul_1
į
1main_level/agent/main/target/global_norm/L2Loss_1L2Losszmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGrad
ń
1main_level/agent/main/target/global_norm/L2Loss_2L2Lossmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1
ł
1main_level/agent/main/target/global_norm/L2Loss_3L2Lossmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad
÷
1main_level/agent/main/target/global_norm/L2Loss_4L2Loss-main_level/agent/main/target/gradients/AddN_1"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape
ß
1main_level/agent/main/target/global_norm/L2Loss_5L2Lossymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
~loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul_1
č
1main_level/agent/main/target/global_norm/L2Loss_6L2Loss}main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGrad
×
1main_level/agent/main/target/global_norm/L2Loss_7L2Lossvmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
}{loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMul_1
į
1main_level/agent/main/target/global_norm/L2Loss_8L2Losszmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGrad
ń
1main_level/agent/main/target/global_norm/L2Loss_9L2Lossmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1
ś
2main_level/agent/main/target/global_norm/L2Loss_10L2Lossmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGrad
ų
2main_level/agent/main/target/global_norm/L2Loss_11L2Loss-main_level/agent/main/target/gradients/AddN_2"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/mul_1_grad/Reshape
Ū
2main_level/agent/main/target/global_norm/L2Loss_12L2Losswmain_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1"/device:GPU:0*
T0*
_class
~|loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1
å
2main_level/agent/main/target/global_norm/L2Loss_13L2Loss{main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGrad"/device:GPU:0*
T0*
_class
loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGrad
Æ
.main_level/agent/main/target/global_norm/stackPack/main_level/agent/main/target/global_norm/L2Loss1main_level/agent/main/target/global_norm/L2Loss_11main_level/agent/main/target/global_norm/L2Loss_21main_level/agent/main/target/global_norm/L2Loss_31main_level/agent/main/target/global_norm/L2Loss_41main_level/agent/main/target/global_norm/L2Loss_51main_level/agent/main/target/global_norm/L2Loss_61main_level/agent/main/target/global_norm/L2Loss_71main_level/agent/main/target/global_norm/L2Loss_81main_level/agent/main/target/global_norm/L2Loss_92main_level/agent/main/target/global_norm/L2Loss_102main_level/agent/main/target/global_norm/L2Loss_112main_level/agent/main/target/global_norm/L2Loss_122main_level/agent/main/target/global_norm/L2Loss_13"/device:GPU:0*
T0*

axis *
N
k
.main_level/agent/main/target/global_norm/ConstConst"/device:GPU:0*
valueB: *
dtype0
Č
,main_level/agent/main/target/global_norm/SumSum.main_level/agent/main/target/global_norm/stack.main_level/agent/main/target/global_norm/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
l
0main_level/agent/main/target/global_norm/Const_1Const"/device:GPU:0*
valueB
 *   @*
dtype0
«
,main_level/agent/main/target/global_norm/mulMul,main_level/agent/main/target/global_norm/Sum0main_level/agent/main/target/global_norm/Const_1"/device:GPU:0*
T0

4main_level/agent/main/target/global_norm/global_normSqrt,main_level/agent/main/target/global_norm/mul"/device:GPU:0*
T0
¦
.main_level/agent/main/target/gradients_1/ShapeShapeEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_1/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_1/FillFill.main_level/agent/main/target/gradients_1/Shape2main_level/agent/main/target/gradients_1/grad_ys_0"/device:GPU:0*
T0*

index_type0
ģ
main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGradBiasAddGrad-main_level/agent/main/target/gradients_1/Fill"/device:GPU:0*
T0*
data_formatNHWC
»
ymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMulMatMul-main_level/agent/main/target/gradients_1/FillImain_level/agent/main/target/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
ø
{main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul_1MatMulDmain_level/agent/main/target/network_0/v_values_head_0/strided_slice-main_level/agent/main/target/gradients_1/Fill"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Õ
xmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_0/add"/device:GPU:0*
T0*
out_type0
ž
main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradxmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/ShapeJmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_2ymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
»
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/ShapeShape*main_level/agent/main/target/network_0/mul"/device:GPU:0*
T0*
out_type0
æ
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape_1Shape,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/SumSummain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/ReshapeReshape\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Sum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Sum_1Summain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Sum_1`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ę
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape_1Shape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/MulMul`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/SumSum\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Mulnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/ReshapeReshape\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Sum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Mul_1Mul*main_level/agent/main/target/network_0/sub`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Sum_1Sum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Mul_1pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Sum_1`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ć
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
ē
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shapebmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0

^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/MulMulbmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape_1.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/SumSum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Mulpmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Sum`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
²
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
ņ
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Mul_1rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ū
dmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Sum_1bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

dmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationdmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1/y_grad/unstack"/device:GPU:0*
T0
å
main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¼
main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
·
main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshapemain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¶
main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ī
|main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

vmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/target/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

xmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_0/observation/submain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
×
jmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/ShapeShape:main_level/agent/main/target/network_0/observation/truediv"/device:GPU:0*
T0*
out_type0
¤
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

zmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shapelmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0

hmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/SumSumvmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMulzmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/ReshapeReshapehmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Sumjmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0

jmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Sum_1Sumvmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMul|main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
hmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/NegNegjmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Sum_1"/device:GPU:0*
T0
÷
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Reshape_1Reshapehmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Neglmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
ß
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/ShapeShape>main_level/agent/main/target/network_0/observation/observation"/device:GPU:0*
T0*
out_type0
Ø
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shapepmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0
æ
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDivRealDivlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Reshape<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0

lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/SumSumpmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
’
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/ReshapeReshapelmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Sumnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ė
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/NegNeg>main_level/agent/main/target/network_0/observation/observation"/device:GPU:0*
T0
Į
rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_1RealDivlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Neg<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
Ē
rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_2RealDivrmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_1<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
ķ
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/mulMullmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Reshapermain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_2"/device:GPU:0*
T0

nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Sum_1Sumlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/mulmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Reshape_1Reshapenmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Sum_1pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
¦
.main_level/agent/main/target/gradients_2/ShapeShapeEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_2/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_2/FillFill.main_level/agent/main/target/gradients_2/Shape2main_level/agent/main/target/gradients_2/grad_ys_0"/device:GPU:0*
T0*

index_type0
¦
.main_level/agent/main/target/gradients_3/ShapeShapeEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_3/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_3/FillFill.main_level/agent/main/target/gradients_3/Shape2main_level/agent/main/target/gradients_3/grad_ys_0"/device:GPU:0*
T0*

index_type0
¦
.main_level/agent/main/target/gradients_4/ShapeShapeEmain_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_4/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_4/FillFill.main_level/agent/main/target/gradients_4/Shape2main_level/agent/main/target/gradients_4/grad_ys_0"/device:GPU:0*
T0*

index_type0

.main_level/agent/main/target/gradients_5/ShapeShape8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_5/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_5/FillFill.main_level/agent/main/target/gradients_5/Shape2main_level/agent/main/target/gradients_5/grad_ys_0"/device:GPU:0*
T0*

index_type0
ņ
jmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mulMul-main_level/agent/main/target/gradients_5/Fill8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0
¾
|main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0

jmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/SumSumjmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul|main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
¤
jmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/subSub-main_level/agent/main/target/gradients_5/Filljmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum"/device:GPU:0*
T0
±
lmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1Muljmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/sub8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0
©
}main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGradBiasAddGradlmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
T0*
data_formatNHWC
ö
wmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMulMatMullmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1Gmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
š
ymain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1MatMul?main_level/agent/main/target/network_1/ppo_head_0/strided_slicelmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Š
smain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_1/add"/device:GPU:0*
T0*
out_type0
ā
~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradsmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/ShapeEmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_2wmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
»
^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/ShapeShape*main_level/agent/main/target/network_1/mul"/device:GPU:0*
T0*
out_type0
æ
`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Shape_1Shape,main_level/agent/main/target/network_1/mul_1"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Shape`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/SumSum~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/ReshapeReshape\main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Sum^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Sum_1Sum~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Sum_1`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ę
`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Shape_1Shape3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Shape`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/MulMul`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Reshape3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/SumSum\main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Mulnmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/ReshapeReshape\main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Sum^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Mul_1Mul*main_level/agent/main/target/network_1/sub`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Reshape"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Sum_1Sum^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Mul_1pmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Sum_1`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ć
bmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0*
out_type0
ē
pmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Shapebmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0

^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/MulMulbmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Reshape_1.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/SumSum^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Mulpmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/ReshapeReshape^main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Sum`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
²
`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/add_grad/Reshape_1"/device:GPU:0*
T0
ņ
`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Sum_1Sum`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Mul_1rmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ū
dmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Reshape_1Reshape`main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Sum_1bmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

dmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1/y_grad/unstackUnpackdmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationdmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/mul_1/y_grad/unstack"/device:GPU:0*
T0
å
main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¼
main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
·
main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshapemain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¶
main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ī
|main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

vmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/target/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

xmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_1/observation/submain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
×
jmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/ShapeShape:main_level/agent/main/target/network_1/observation/truediv"/device:GPU:0*
T0*
out_type0
¤
lmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

zmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Shapelmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Shape_1"/device:GPU:0*
T0

hmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/SumSumvmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMulzmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
lmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/ReshapeReshapehmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Sumjmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0

jmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Sum_1Sumvmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMul|main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
ó
hmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/NegNegjmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Sum_1"/device:GPU:0*
T0
÷
nmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Reshape_1Reshapehmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Neglmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
ß
nmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/ShapeShape>main_level/agent/main/target/network_0/observation/observation"/device:GPU:0*
T0*
out_type0
Ø
pmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0

~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Shapepmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Shape_1"/device:GPU:0*
T0
æ
pmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/RealDivRealDivlmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Reshape<main_level/agent/main/target/network_1/observation/truediv/y"/device:GPU:0*
T0

lmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/SumSumpmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/RealDiv~main_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
’
pmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/ReshapeReshapelmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Sumnmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Shape"/device:GPU:0*
T0*
Tshape0
Ė
lmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/NegNeg>main_level/agent/main/target/network_0/observation/observation"/device:GPU:0*
T0
Į
rmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/RealDiv_1RealDivlmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Neg<main_level/agent/main/target/network_1/observation/truediv/y"/device:GPU:0*
T0
Ē
rmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/RealDiv_2RealDivrmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/RealDiv_1<main_level/agent/main/target/network_1/observation/truediv/y"/device:GPU:0*
T0
ķ
lmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/mulMullmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/sub_grad/Reshapermain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/RealDiv_2"/device:GPU:0*
T0

nmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Sum_1Sumlmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/mulmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0

rmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Reshape_1Reshapenmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Sum_1pmain_level/agent/main/target/gradients_5/main_level/agent/main/target/network_1/observation/truediv_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

.main_level/agent/main/target/gradients_6/ShapeShape8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_6/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_6/FillFill.main_level/agent/main/target/gradients_6/Shape2main_level/agent/main/target/gradients_6/grad_ys_0"/device:GPU:0*
T0*

index_type0

.main_level/agent/main/target/gradients_7/ShapeShape8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_7/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_7/FillFill.main_level/agent/main/target/gradients_7/Shape2main_level/agent/main/target/gradients_7/grad_ys_0"/device:GPU:0*
T0*

index_type0

.main_level/agent/main/target/gradients_8/ShapeShape8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_8/grad_ys_0Const"/device:GPU:0*
valueB
 *  ?*
dtype0
Ć
-main_level/agent/main/target/gradients_8/FillFill.main_level/agent/main/target/gradients_8/Shape2main_level/agent/main/target/gradients_8/grad_ys_0"/device:GPU:0*
T0*

index_type0
}
4main_level/agent/main/target/output_gradient_weightsPlaceholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

6main_level/agent/main/target/output_gradient_weights_1Placeholder"/device:GPU:0*
shape:’’’’’’’’’*
dtype0

2main_level/agent/main/target/gradients_9/grad_ys_0Identity4main_level/agent/main/target/output_gradient_weights"/device:GPU:0*
T0
ń
main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/output/BiasAdd_grad/BiasAddGradBiasAddGrad2main_level/agent/main/target/gradients_9/grad_ys_0"/device:GPU:0*
T0*
data_formatNHWC
Ą
ymain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMulMatMul2main_level/agent/main/target/gradients_9/grad_ys_0Imain_level/agent/main/target/network_0/v_values_head_0/output/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
½
{main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul_1MatMulDmain_level/agent/main/target/network_0/v_values_head_0/strided_slice2main_level/agent/main/target/gradients_9/grad_ys_0"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Õ
xmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_0/add"/device:GPU:0*
T0*
out_type0
ž
main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradxmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/ShapeJmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stackLmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_1Lmain_level/agent/main/target/network_0/v_values_head_0/strided_slice/stack_2ymain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/output/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
»
^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/ShapeShape*main_level/agent/main/target/network_0/mul"/device:GPU:0*
T0*
out_type0
æ
`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Shape_1Shape,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Shape`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/SumSummain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/ReshapeReshape\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Sum^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Sum_1Summain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/v_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Sum_1`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ę
`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Shape_1Shape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
į
nmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Shape`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0

\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/MulMul`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Reshape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
ę
\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/SumSum\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Mulnmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ļ
`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/ReshapeReshape\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Sum^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Mul_1Mul*main_level/agent/main/target/network_0/sub`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Reshape"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Sum_1Sum^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Mul_1pmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Sum_1`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ć
bmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
ē
pmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Shapebmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0

^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/MulMulbmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Reshape_1.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
ģ
^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/SumSum^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Mulpmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Õ
bmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Sum`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
²
`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
ņ
`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Mul_1rmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ū
dmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Sum_1bmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
Ż
\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/sub_grad/NegNeg`main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_grad/Reshape"/device:GPU:0*
T0

dmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

-main_level/agent/main/target/gradients_9/AddNAddNbmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Reshape\main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/sub_grad/Neg"/device:GPU:0*
T0*u
_classk
igloc:@main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1_grad/Reshape*
N

main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activationdmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/mul_1/y_grad/unstack"/device:GPU:0*
T0
å
main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¼
main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
·
main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshapemain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

~main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¶
main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul~main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Ī
|main_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

vmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/target/network_0/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

xmain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_0/observation/submain_level/agent/main/target/gradients_9/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

3main_level/agent/main/target/gradients_10/grad_ys_0Identity6main_level/agent/main/target/output_gradient_weights_1"/device:GPU:0*
T0
ł
kmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mulMul3main_level/agent/main/target/gradients_10/grad_ys_08main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0
æ
}main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
’’’’’’’’’*
dtype0

kmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/SumSumkmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul}main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
¬
kmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/subSub3main_level/agent/main/target/gradients_10/grad_ys_0kmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/Sum"/device:GPU:0*
T0
³
mmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1Mulkmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/sub8main_level/agent/main/target/network_1/ppo_head_0/policy"/device:GPU:0*
T0
«
~main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/BiasAdd_grad/BiasAddGradBiasAddGradmmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
T0*
data_formatNHWC
ų
xmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMulMatMulmmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1Gmain_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
ņ
zmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul_1MatMul?main_level/agent/main/target/network_1/ppo_head_0/strided_slicemmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_grad/mul_1"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
Ń
tmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_1/add"/device:GPU:0*
T0*
out_type0
å
main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradStridedSliceGradtmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/ShapeEmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stackGmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_1Gmain_level/agent/main/target/network_1/ppo_head_0/strided_slice/stack_2xmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/policy_fc/MatMul_grad/MatMul"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
¼
_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/ShapeShape*main_level/agent/main/target/network_1/mul"/device:GPU:0*
T0*
out_type0
Ą
amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Shape_1Shape,main_level/agent/main/target/network_1/mul_1"/device:GPU:0*
T0*
out_type0
ä
omain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Shapeamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Shape_1"/device:GPU:0*
T0

]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/SumSummain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradomain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ņ
amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/ReshapeReshape]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Sum_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0

_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Sum_1Summain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/ppo_head_0/strided_slice_grad/StridedSliceGradqmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ų
cmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Reshape_1Reshape_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Sum_1amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ē
amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Shape_1Shape3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0*
out_type0
ä
omain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Shapeamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Shape_1"/device:GPU:0*
T0

]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/MulMulamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Reshape3main_level/agent/main/target/network_1/StopGradient"/device:GPU:0*
T0
é
]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/SumSum]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Mulomain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ņ
amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/ReshapeReshape]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Sum_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0

_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Mul_1Mul*main_level/agent/main/target/network_1/subamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Reshape"/device:GPU:0*
T0
ļ
_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Sum_1Sum_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Mul_1qmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ų
cmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Reshape_1Reshape_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Sum_1amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0

amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
Ä
cmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0*
out_type0
ź
qmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Shapecmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0

_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/MulMulcmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Reshape_1.main_level/agent/main/target/network_1/mul_1/y"/device:GPU:0*
T0
ļ
_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/SumSum_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Mulqmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ų
cmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/ReshapeReshape_main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Sumamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
“
amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/readcmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/add_grad/Reshape_1"/device:GPU:0*
T0
õ
amain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Sum_1Sumamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Mul_1smain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
Ž
emain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Reshape_1Reshapeamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Sum_1cmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
ß
]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/sub_grad/NegNegamain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_grad/Reshape"/device:GPU:0*
T0

emain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1/y_grad/unstackUnpackemain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 

.main_level/agent/main/target/gradients_10/AddNAddNcmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Reshape]main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/sub_grad/Neg"/device:GPU:0*
T0*v
_classl
jhloc:@main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1_grad/Reshape*
N

main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGrademain_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activationemain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/mul_1/y_grad/unstack"/device:GPU:0*
T0
ē
main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC
¾
main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGradQmain_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
¹
main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMul_1MatMulJmain_level/agent/main/target/network_1/observation/Flatten/flatten/Reshapemain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(

main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
¹
main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/ReshapeReshapemain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/MatMul_grad/MatMulmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0

main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradTanhGradZmain_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activationmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Flatten/flatten/Reshape_grad/Reshape"/device:GPU:0*
T0
Š
}main_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Dense_0/BiasAdd_grad/BiasAddGradBiasAddGradmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
T0*
data_formatNHWC

wmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMulMatMulmain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGradFmain_level/agent/main/target/network_1/observation/Dense_0/kernel/read"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 

ymain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/Dense_0/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_1/observation/submain_level/agent/main/target/gradients_10/main_level/agent/main/target/network_1/observation/BatchnormActivationDropout_1_activation_grad/TanhGrad"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
e
,main_level/agent/main/target/AssignAdd/valueConst"/device:GPU:0*
value	B	 R*
dtype0	
ó
&main_level/agent/main/target/AssignAdd	AssignAdd(main_level/agent/main/online/global_step,main_level/agent/main/target/AssignAdd/value"/device:GPU:0*
use_locking( *
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
ā+
!main_level/agent/main/target/initNoOp0^main_level/agent/main/online/beta1_power/Assign0^main_level/agent/main/online/beta2_power/Assign0^main_level/agent/main/online/global_step/Assignr^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/Adam_1/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/Adam_1/Assigni^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/bias/Adam_1/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam/Assignm^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/Dense_0/kernel/Adam_1/Assignl^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam/Assignn^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/bias/Adam_1/Assignn^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam/Assignp^main_level/agent/main/online/main_level/agent/main/online/network_0/v_values_head_0/output/kernel/Adam_1/Assignr^main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/Adam_1/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/Adam_1/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/Adam_1/Assigni^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/bias/Adam_1/Assignk^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam/Assignm^main_level/agent/main/online/main_level/agent/main/online/network_1/observation/Dense_0/kernel/Adam_1/Assignj^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam/Assignl^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/Adam_1/Assignl^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam/Assignn^main_level/agent/main/online/main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/Adam_1/AssignP^main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/AssignR^main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/bias/AssignT^main_level/agent/main/online/network_0/middleware_fc_embedder/Dense_0/kernel/AssignG^main_level/agent/main/online/network_0/observation/Dense_0/bias/AssignI^main_level/agent/main/online/network_0/observation/Dense_0/kernel/AssignJ^main_level/agent/main/online/network_0/v_values_head_0/output/bias/AssignL^main_level/agent/main/online/network_0/v_values_head_0/output/kernel/AssignP^main_level/agent/main/online/network_1/gradients_from_head_0-0_rescalers/AssignR^main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/bias/AssignT^main_level/agent/main/online/network_1/middleware_fc_embedder/Dense_0/kernel/AssignG^main_level/agent/main/online/network_1/observation/Dense_0/bias/AssignI^main_level/agent/main/online/network_1/observation/Dense_0/kernel/AssignH^main_level/agent/main/online/network_1/ppo_head_0/policy_fc/bias/AssignJ^main_level/agent/main/online/network_1/ppo_head_0/policy_fc/kernel/AssignP^main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/AssignR^main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/bias/AssignT^main_level/agent/main/target/network_0/middleware_fc_embedder/Dense_0/kernel/AssignG^main_level/agent/main/target/network_0/observation/Dense_0/bias/AssignI^main_level/agent/main/target/network_0/observation/Dense_0/kernel/AssignJ^main_level/agent/main/target/network_0/v_values_head_0/output/bias/AssignL^main_level/agent/main/target/network_0/v_values_head_0/output/kernel/AssignP^main_level/agent/main/target/network_1/gradients_from_head_0-0_rescalers/AssignR^main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/bias/AssignT^main_level/agent/main/target/network_1/middleware_fc_embedder/Dense_0/kernel/AssignG^main_level/agent/main/target/network_1/observation/Dense_0/bias/AssignI^main_level/agent/main/target/network_1/observation/Dense_0/kernel/AssignH^main_level/agent/main/target/network_1/ppo_head_0/policy_fc/bias/AssignJ^main_level/agent/main/target/network_1/ppo_head_0/policy_fc/kernel/Assign"/device:GPU:0
ü
#main_level/agent/main/target/init_1NoOp-^main_level/agent/main/online/Variable/Assign7^main_level/agent/main/online/network_0/Variable/Assign7^main_level/agent/main/online/network_1/Variable/Assign-^main_level/agent/main/target/Variable/Assign7^main_level/agent/main/target/network_0/Variable/Assign7^main_level/agent/main/target/network_1/Variable/Assign"/device:GPU:0

'main_level/agent/main/target/group_depsNoOp"^main_level/agent/main/target/init$^main_level/agent/main/target/init_1"/device:GPU:0"&