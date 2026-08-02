module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["diloco"=1, "data"=1, "stage"=1, "fsdp"=1, "fsdp_transpose"=1, "context"=1, "context_autoregressive"=1, "tensor"=1, "tensor_sequence"=1, "expert"=1, "autoregressive"=1]> {stablehlo.mesh = {axes = [{name = "diloco", size = 1 : i64}, {name = "data", size = 1 : i64}, {name = "stage", size = 1 : i64}, {name = "fsdp", size = 1 : i64}, {name = "fsdp_transpose", size = 1 : i64}, {name = "context", size = 1 : i64}, {name = "context_autoregressive", size = 1 : i64}, {name = "tensor", size = 1 : i64}, {name = "tensor_sequence", size = 1 : i64}, {name = "expert", size = 1 : i64}, {name = "autoregressive", size = 1 : i64}]}}
  func.func public @main() -> (tensor<1024xf32> {jax.result_info = "result['model']['decoder']['decoder_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<ui32> {jax.result_info = "result['model']['decoder']['dropout']['rngs']['aqt']['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<2xui32> {jax.result_info = "result['model']['decoder']['dropout']['rngs']['aqt']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<ui32> {jax.result_info = "result['model']['decoder']['dropout']['rngs']['dropout']['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<2xui32> {jax.result_info = "result['model']['decoder']['dropout']['rngs']['dropout']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<ui32> {jax.result_info = "result['model']['decoder']['dropout']['rngs']['params']['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<2xui32> {jax.result_info = "result['model']['decoder']['dropout']['rngs']['params']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28xui32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['dropout']['rngs']['aqt']['count'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28x2xui32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['dropout']['rngs']['aqt']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<28xui32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['dropout']['rngs']['dropout']['count'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28x2xui32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['dropout']['rngs']['dropout']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<28xui32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['dropout']['rngs']['params']['count'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28x2xui32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['dropout']['rngs']['params']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x3072xf32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['wi_0']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<1024x28x3072xf32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['wi_1']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<3072x28x1024xf32> {jax.result_info = "result['model']['decoder']['layers']['mlp']['wo']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<1024x28xf32> {jax.result_info = "result['model']['decoder']['layers']['post_self_attention_layer_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28xf32> {jax.result_info = "result['model']['decoder']['layers']['pre_self_attention_layer_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<28xui32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['attention_op']['rngs']['aqt']['count'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28x2xui32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['attention_op']['rngs']['aqt']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<28xui32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['attention_op']['rngs']['dropout']['count'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28x2xui32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['attention_op']['rngs']['dropout']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<28xui32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['attention_op']['rngs']['params']['count'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<28x2xui32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['attention_op']['rngs']['params']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x8x128xf32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['key']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<128x28xf32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['key_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<16x28x128x1024xf32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['out']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<1024x28x16x128xf32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['query']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<128x28xf32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['query_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x8x128xf32> {jax.result_info = "result['model']['decoder']['layers']['self_attention']['value']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<ui32> {jax.result_info = "result['model']['decoder']['rngs']['aqt']['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<2xui32> {jax.result_info = "result['model']['decoder']['rngs']['aqt']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<ui32> {jax.result_info = "result['model']['decoder']['rngs']['dropout']['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<2xui32> {jax.result_info = "result['model']['decoder']['rngs']['dropout']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<ui32> {jax.result_info = "result['model']['decoder']['rngs']['params']['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<2xui32> {jax.result_info = "result['model']['decoder']['rngs']['params']['key'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<151936x1024xf32> {jax.result_info = "result['model']['token_embedder']['embedding'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<i32> {jax.result_info = "result['optimizer']['opt_state'][0]['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['decoder_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<1024x28x3072xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['mlp']['wi_0']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<1024x28x3072xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['mlp']['wi_1']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<3072x28x1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['mlp']['wo']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<1024x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['post_self_attention_layer_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['pre_self_attention_layer_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x8x128xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['self_attention']['key']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<128x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['self_attention']['key_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<16x28x128x1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['self_attention']['out']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<1024x28x16x128xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['self_attention']['query']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<128x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['self_attention']['query_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x8x128xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['decoder']['layers']['self_attention']['value']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<151936x1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['mu']['token_embedder']['embedding'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['decoder_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<1024x28x3072xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['mlp']['wi_0']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<1024x28x3072xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['mlp']['wi_1']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<3072x28x1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['mlp']['wo']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, tensor<1024x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['post_self_attention_layer_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['pre_self_attention_layer_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x8x128xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['self_attention']['key']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<128x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['self_attention']['key_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<16x28x128x1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['self_attention']['out']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<1024x28x16x128xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['self_attention']['query']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<128x28xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['self_attention']['query_norm']['scale'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1024x28x8x128xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['decoder']['layers']['self_attention']['value']['kernel'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, tensor<151936x1024xf32> {jax.result_info = "result['optimizer']['opt_state'][0]['nu']['token_embedder']['embedding'].value", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<i32> {jax.result_info = "result['optimizer']['opt_state'][2]['count'].value", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<ui32> {jax.result_info = "result['optimizer']['step'].value", sdy.sharding = #sdy.sharding<@mesh, []>}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<28> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<2xui32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_3 = stablehlo.constant dense<2> : tensor<ui32>
    %c_4 = stablehlo.constant dense<1> : tensor<ui32>
    %c_5 = stablehlo.constant dense<0> : tensor<ui32>
    %c_6 = stablehlo.constant dense<-1> : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %c_8 = stablehlo.constant dense<32> : tensor<i32>
    %0 = stablehlo.shift_right_logical %c_7, %c_8 : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %3 = stablehlo.and %c_7, %c_6 : tensor<i32>
    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<ui32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %6 = stablehlo.concatenate %2, %5, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %7 = call @_threefry_fold_in(%6, %c_5) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %8 = call @_threefry_fold_in(%6, %c_4) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %9 = call @_threefry_fold_in(%6, %c_3) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %10 = call @_threefry_fold_in(%7, %c_5) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %11 = stablehlo.add %c_5, %c_4 : tensor<ui32>
    %12 = call @_normal(%10) : (tensor<2xui32>) -> tensor<151936x1024xf32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<151936x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<151936x1024xf32>
    %15 = call @_threefry_fold_in(%7, %11) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %16 = stablehlo.add %11, %c_4 : tensor<ui32>
    %17 = call @_threefry_fold_in(%8, %c_5) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %18 = stablehlo.add %c_5, %c_4 : tensor<ui32>
    %19 = call @_threefry_fold_in(%9, %c_5) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %20 = stablehlo.add %c_5, %c_4 : tensor<ui32>
    %21 = stablehlo.add %16, %c_4 : tensor<ui32>
    %22 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %23 = call @_threefry_fold_in(%7, %21) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %24 = stablehlo.add %21, %c_4 : tensor<ui32>
    %25 = call @_threefry_split(%23) : (tensor<2xui32>) -> tensor<28x2xui32>
    %26 = call @_threefry_fold_in(%8, %18) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %27 = stablehlo.add %18, %c_4 : tensor<ui32>
    %28 = call @_threefry_split(%26) : (tensor<2xui32>) -> tensor<28x2xui32>
    %29 = call @_threefry_fold_in(%9, %20) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %30 = stablehlo.add %20, %c_4 : tensor<ui32>
    %31 = call @_threefry_split(%29) : (tensor<2xui32>) -> tensor<28x2xui32>
    %32 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %33 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %34 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %35 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024x3072xf32>
    %36 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024x3072xf32>
    %37 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x3072x1024xf32>
    %38 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024xf32>
    %39 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024x8x128xf32>
    %41 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x128xf32>
    %42 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x16x128x1024xf32>
    %43 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024x16x128xf32>
    %44 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x128xf32>
    %45 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<28x1024x8x128xf32>
    %46 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %47 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<2xui32>) -> tensor<28x2xui32>
    %48 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %49 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<2xui32>) -> tensor<28x2xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %51 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<2xui32>) -> tensor<28x2xui32>
    %52 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %53 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %54 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %55:27 = stablehlo.while(%iterArg = %34, %iterArg_9 = %31, %iterArg_10 = %33, %iterArg_11 = %28, %iterArg_12 = %32, %iterArg_13 = %25, %iterArg_14 = %c_7, %iterArg_15 = %35, %iterArg_16 = %36, %iterArg_17 = %37, %iterArg_18 = %38, %iterArg_19 = %39, %iterArg_20 = %40, %iterArg_21 = %41, %iterArg_22 = %42, %iterArg_23 = %43, %iterArg_24 = %44, %iterArg_25 = %45, %iterArg_26 = %46, %iterArg_27 = %47, %iterArg_28 = %48, %iterArg_29 = %49, %iterArg_30 = %50, %iterArg_31 = %51, %iterArg_32 = %52, %iterArg_33 = %53, %iterArg_34 = %54) : tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<i32>, tensor<28x1024x3072xf32>, tensor<28x1024x3072xf32>, tensor<28x3072x1024xf32>, tensor<28x1024xf32>, tensor<28x1024xf32>, tensor<28x1024x8x128xf32>, tensor<28x128xf32>, tensor<28x16x128x1024xf32>, tensor<28x1024x16x128xf32>, tensor<28x128xf32>, tensor<28x1024x8x128xf32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28xui32>, tensor<28xui32>
    cond {
      %93 = stablehlo.compare LT, %iterArg_14, %c_0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %93 : tensor<i1>
    } do {
      %93 = func.call @dynamic_index_in_dim(%iterArg, %iterArg_14) : (tensor<28xui32>, tensor<i32>) -> tensor<ui32>
      %94 = func.call @dynamic_index_in_dim_2(%iterArg_9, %iterArg_14) : (tensor<28x2xui32>, tensor<i32>) -> tensor<2xui32>
      %95 = func.call @dynamic_index_in_dim(%iterArg_10, %iterArg_14) : (tensor<28xui32>, tensor<i32>) -> tensor<ui32>
      %96 = func.call @dynamic_index_in_dim_2(%iterArg_11, %iterArg_14) : (tensor<28x2xui32>, tensor<i32>) -> tensor<2xui32>
      %97 = func.call @dynamic_index_in_dim(%iterArg_12, %iterArg_14) : (tensor<28xui32>, tensor<i32>) -> tensor<ui32>
      %98 = func.call @dynamic_index_in_dim_2(%iterArg_13, %iterArg_14) : (tensor<28x2xui32>, tensor<i32>) -> tensor<2xui32>
      %99:20 = func.call @closed_call(%93, %94, %95, %96, %97, %98) : (tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>) -> (tensor<1024x3072xf32>, tensor<1024x3072xf32>, tensor<3072x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x8x128xf32>, tensor<128xf32>, tensor<16x128x1024xf32>, tensor<1024x16x128xf32>, tensor<128xf32>, tensor<1024x8x128xf32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>)
      %100 = func.call @dynamic_update_index_in_dim(%iterArg_15, %99#0, %iterArg_14) : (tensor<28x1024x3072xf32>, tensor<1024x3072xf32>, tensor<i32>) -> tensor<28x1024x3072xf32>
      %101 = func.call @dynamic_update_index_in_dim(%iterArg_16, %99#1, %iterArg_14) : (tensor<28x1024x3072xf32>, tensor<1024x3072xf32>, tensor<i32>) -> tensor<28x1024x3072xf32>
      %102 = func.call @dynamic_update_index_in_dim_21(%iterArg_17, %99#2, %iterArg_14) : (tensor<28x3072x1024xf32>, tensor<3072x1024xf32>, tensor<i32>) -> tensor<28x3072x1024xf32>
      %103 = func.call @dynamic_update_index_in_dim_22(%iterArg_18, %99#3, %iterArg_14) : (tensor<28x1024xf32>, tensor<1024xf32>, tensor<i32>) -> tensor<28x1024xf32>
      %104 = func.call @dynamic_update_index_in_dim_22(%iterArg_19, %99#4, %iterArg_14) : (tensor<28x1024xf32>, tensor<1024xf32>, tensor<i32>) -> tensor<28x1024xf32>
      %105 = func.call @dynamic_update_index_in_dim_23(%iterArg_20, %99#5, %iterArg_14) : (tensor<28x1024x8x128xf32>, tensor<1024x8x128xf32>, tensor<i32>) -> tensor<28x1024x8x128xf32>
      %106 = func.call @dynamic_update_index_in_dim_24(%iterArg_21, %99#6, %iterArg_14) : (tensor<28x128xf32>, tensor<128xf32>, tensor<i32>) -> tensor<28x128xf32>
      %107 = func.call @dynamic_update_index_in_dim_25(%iterArg_22, %99#7, %iterArg_14) : (tensor<28x16x128x1024xf32>, tensor<16x128x1024xf32>, tensor<i32>) -> tensor<28x16x128x1024xf32>
      %108 = func.call @dynamic_update_index_in_dim_26(%iterArg_23, %99#8, %iterArg_14) : (tensor<28x1024x16x128xf32>, tensor<1024x16x128xf32>, tensor<i32>) -> tensor<28x1024x16x128xf32>
      %109 = func.call @dynamic_update_index_in_dim_24(%iterArg_24, %99#9, %iterArg_14) : (tensor<28x128xf32>, tensor<128xf32>, tensor<i32>) -> tensor<28x128xf32>
      %110 = func.call @dynamic_update_index_in_dim_23(%iterArg_25, %99#10, %iterArg_14) : (tensor<28x1024x8x128xf32>, tensor<1024x8x128xf32>, tensor<i32>) -> tensor<28x1024x8x128xf32>
      %111 = func.call @dynamic_update_index_in_dim_27(%iterArg_26, %99#11, %iterArg_14) : (tensor<28xui32>, tensor<ui32>, tensor<i32>) -> tensor<28xui32>
      %112 = func.call @dynamic_update_index_in_dim_28(%iterArg_27, %99#12, %iterArg_14) : (tensor<28x2xui32>, tensor<2xui32>, tensor<i32>) -> tensor<28x2xui32>
      %113 = func.call @dynamic_update_index_in_dim_27(%iterArg_28, %99#13, %iterArg_14) : (tensor<28xui32>, tensor<ui32>, tensor<i32>) -> tensor<28xui32>
      %114 = func.call @dynamic_update_index_in_dim_28(%iterArg_29, %99#14, %iterArg_14) : (tensor<28x2xui32>, tensor<2xui32>, tensor<i32>) -> tensor<28x2xui32>
      %115 = func.call @dynamic_update_index_in_dim_27(%iterArg_30, %99#15, %iterArg_14) : (tensor<28xui32>, tensor<ui32>, tensor<i32>) -> tensor<28xui32>
      %116 = func.call @dynamic_update_index_in_dim_28(%iterArg_31, %99#16, %iterArg_14) : (tensor<28x2xui32>, tensor<2xui32>, tensor<i32>) -> tensor<28x2xui32>
      %117 = func.call @dynamic_update_index_in_dim_27(%iterArg_32, %99#17, %iterArg_14) : (tensor<28xui32>, tensor<ui32>, tensor<i32>) -> tensor<28xui32>
      %118 = func.call @dynamic_update_index_in_dim_27(%iterArg_33, %99#18, %iterArg_14) : (tensor<28xui32>, tensor<ui32>, tensor<i32>) -> tensor<28xui32>
      %119 = func.call @dynamic_update_index_in_dim_27(%iterArg_34, %99#19, %iterArg_14) : (tensor<28xui32>, tensor<ui32>, tensor<i32>) -> tensor<28xui32>
      %120 = stablehlo.add %iterArg_14, %c : tensor<i32>
      stablehlo.return %iterArg, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %120, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119 : tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<i32>, tensor<28x1024x3072xf32>, tensor<28x1024x3072xf32>, tensor<28x3072x1024xf32>, tensor<28x1024xf32>, tensor<28x1024xf32>, tensor<28x1024x8x128xf32>, tensor<28x128xf32>, tensor<28x16x128x1024xf32>, tensor<28x1024x16x128xf32>, tensor<28x128xf32>, tensor<28x1024x8x128xf32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28xui32>, tensor<28xui32>
    }
    %56 = stablehlo.transpose %55#7, dims = [1, 0, 2] : (tensor<28x1024x3072xf32>) -> tensor<1024x28x3072xf32>
    %57 = stablehlo.transpose %55#8, dims = [1, 0, 2] : (tensor<28x1024x3072xf32>) -> tensor<1024x28x3072xf32>
    %58 = stablehlo.transpose %55#9, dims = [1, 0, 2] : (tensor<28x3072x1024xf32>) -> tensor<3072x28x1024xf32>
    %59 = stablehlo.transpose %55#10, dims = [1, 0] : (tensor<28x1024xf32>) -> tensor<1024x28xf32>
    %60 = stablehlo.transpose %55#11, dims = [1, 0] : (tensor<28x1024xf32>) -> tensor<1024x28xf32>
    %61 = stablehlo.transpose %55#12, dims = [1, 0, 2, 3] : (tensor<28x1024x8x128xf32>) -> tensor<1024x28x8x128xf32>
    %62 = stablehlo.transpose %55#13, dims = [1, 0] : (tensor<28x128xf32>) -> tensor<128x28xf32>
    %63 = stablehlo.transpose %55#14, dims = [1, 0, 2, 3] : (tensor<28x16x128x1024xf32>) -> tensor<16x28x128x1024xf32>
    %64 = stablehlo.transpose %55#15, dims = [1, 0, 2, 3] : (tensor<28x1024x16x128xf32>) -> tensor<1024x28x16x128xf32>
    %65 = stablehlo.transpose %55#16, dims = [1, 0] : (tensor<28x128xf32>) -> tensor<128x28xf32>
    %66 = stablehlo.transpose %55#17, dims = [1, 0, 2, 3] : (tensor<28x1024x8x128xf32>) -> tensor<1024x28x8x128xf32>
    %67 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %68 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x3072xf32>
    %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x3072xf32>
    %70 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3072x28x1024xf32>
    %71 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28xf32>
    %72 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28xf32>
    %73 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x8x128xf32>
    %74 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28xf32>
    %75 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x28x128x1024xf32>
    %76 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x16x128xf32>
    %77 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28xf32>
    %78 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x8x128xf32>
    %79 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<151936x1024xf32>
    %80 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %81 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x3072xf32>
    %82 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x3072xf32>
    %83 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3072x28x1024xf32>
    %84 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28xf32>
    %85 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28xf32>
    %86 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x8x128xf32>
    %87 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28xf32>
    %88 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x28x128x1024xf32>
    %89 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x16x128xf32>
    %90 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28xf32>
    %91 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x28x8x128xf32>
    %92 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<151936x1024xf32>
    return %22, %c_5, %19, %c_5, %17, %c_5, %15, %55#18, %55#19, %55#20, %55#21, %55#22, %55#23, %56, %57, %58, %59, %60, %55#24, %31, %55#25, %28, %55#26, %25, %61, %62, %63, %64, %65, %66, %30, %9, %27, %8, %24, %7, %14, %c_7, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %c_7, %c_5 : tensor<1024xf32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<1024x28x3072xf32>, tensor<1024x28x3072xf32>, tensor<3072x28x1024xf32>, tensor<1024x28xf32>, tensor<1024x28xf32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<28xui32>, tensor<28x2xui32>, tensor<1024x28x8x128xf32>, tensor<128x28xf32>, tensor<16x28x128x1024xf32>, tensor<1024x28x16x128xf32>, tensor<128x28xf32>, tensor<1024x28x8x128xf32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<151936x1024xf32>, tensor<i32>, tensor<1024xf32>, tensor<1024x28x3072xf32>, tensor<1024x28x3072xf32>, tensor<3072x28x1024xf32>, tensor<1024x28xf32>, tensor<1024x28xf32>, tensor<1024x28x8x128xf32>, tensor<128x28xf32>, tensor<16x28x128x1024xf32>, tensor<1024x28x16x128xf32>, tensor<128x28xf32>, tensor<1024x28x8x128xf32>, tensor<151936x1024xf32>, tensor<1024xf32>, tensor<1024x28x3072xf32>, tensor<1024x28x3072xf32>, tensor<3072x28x1024xf32>, tensor<1024x28xf32>, tensor<1024x28xf32>, tensor<1024x28x8x128xf32>, tensor<128x28xf32>, tensor<16x28x128x1024xf32>, tensor<1024x28x16x128xf32>, tensor<128x28xf32>, tensor<1024x28x8x128xf32>, tensor<151936x1024xf32>, tensor<i32>, tensor<ui32>
  }
  func.func private @_threefry_fold_in(%arg0: tensor<2xui32>, %arg1: tensor<ui32>) -> tensor<2xui32> {
    %c = stablehlo.constant dense<4294967295> : tensor<ui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %0 = stablehlo.shift_right_logical %arg1, %c_0 : tensor<ui32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %2 = stablehlo.and %arg1, %c : tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = stablehlo.slice %4 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.slice %4 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %11:2 = call @threefry2x32(%6, %8, %9, %10) : (tensor<ui32>, tensor<ui32>, tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>, tensor<1xui32>)
    %12 = stablehlo.concatenate %11#0, %11#1, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    return %12 : tensor<2xui32>
  }
  func.func private @threefry2x32(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<1xui32>, %arg3: tensor<1xui32>) -> (tensor<1xui32>, tensor<1xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<1xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<1xui32>
    %6 = stablehlo.add %3, %5 : tensor<1xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<1xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<1xui32>
    %11 = stablehlo.or %8, %10 : tensor<1xui32>
    %12 = stablehlo.xor %6, %11 : tensor<1xui32>
    %13 = stablehlo.add %6, %12 : tensor<1xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<1xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<1xui32>
    %18 = stablehlo.or %15, %17 : tensor<1xui32>
    %19 = stablehlo.xor %13, %18 : tensor<1xui32>
    %20 = stablehlo.add %13, %19 : tensor<1xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<1xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<1xui32>
    %25 = stablehlo.or %22, %24 : tensor<1xui32>
    %26 = stablehlo.xor %20, %25 : tensor<1xui32>
    %27 = stablehlo.add %20, %26 : tensor<1xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<1xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<1xui32>
    %32 = stablehlo.or %29, %31 : tensor<1xui32>
    %33 = stablehlo.xor %27, %32 : tensor<1xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %35 = stablehlo.add %27, %34 : tensor<1xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %37 = stablehlo.add %33, %36 : tensor<1xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %39 = stablehlo.add %37, %38 : tensor<1xui32>
    %40 = stablehlo.add %35, %39 : tensor<1xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<1xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<1xui32>
    %45 = stablehlo.or %42, %44 : tensor<1xui32>
    %46 = stablehlo.xor %40, %45 : tensor<1xui32>
    %47 = stablehlo.add %40, %46 : tensor<1xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<1xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<1xui32>
    %52 = stablehlo.or %49, %51 : tensor<1xui32>
    %53 = stablehlo.xor %47, %52 : tensor<1xui32>
    %54 = stablehlo.add %47, %53 : tensor<1xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<1xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<1xui32>
    %59 = stablehlo.or %56, %58 : tensor<1xui32>
    %60 = stablehlo.xor %54, %59 : tensor<1xui32>
    %61 = stablehlo.add %54, %60 : tensor<1xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<1xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<1xui32>
    %66 = stablehlo.or %63, %65 : tensor<1xui32>
    %67 = stablehlo.xor %61, %66 : tensor<1xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %69 = stablehlo.add %61, %68 : tensor<1xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %71 = stablehlo.add %67, %70 : tensor<1xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %73 = stablehlo.add %71, %72 : tensor<1xui32>
    %74 = stablehlo.add %69, %73 : tensor<1xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<1xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<1xui32>
    %79 = stablehlo.or %76, %78 : tensor<1xui32>
    %80 = stablehlo.xor %74, %79 : tensor<1xui32>
    %81 = stablehlo.add %74, %80 : tensor<1xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<1xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<1xui32>
    %86 = stablehlo.or %83, %85 : tensor<1xui32>
    %87 = stablehlo.xor %81, %86 : tensor<1xui32>
    %88 = stablehlo.add %81, %87 : tensor<1xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<1xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<1xui32>
    %93 = stablehlo.or %90, %92 : tensor<1xui32>
    %94 = stablehlo.xor %88, %93 : tensor<1xui32>
    %95 = stablehlo.add %88, %94 : tensor<1xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<1xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<1xui32>
    %100 = stablehlo.or %97, %99 : tensor<1xui32>
    %101 = stablehlo.xor %95, %100 : tensor<1xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %103 = stablehlo.add %95, %102 : tensor<1xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %105 = stablehlo.add %101, %104 : tensor<1xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %107 = stablehlo.add %105, %106 : tensor<1xui32>
    %108 = stablehlo.add %103, %107 : tensor<1xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<1xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<1xui32>
    %113 = stablehlo.or %110, %112 : tensor<1xui32>
    %114 = stablehlo.xor %108, %113 : tensor<1xui32>
    %115 = stablehlo.add %108, %114 : tensor<1xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<1xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<1xui32>
    %120 = stablehlo.or %117, %119 : tensor<1xui32>
    %121 = stablehlo.xor %115, %120 : tensor<1xui32>
    %122 = stablehlo.add %115, %121 : tensor<1xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<1xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<1xui32>
    %127 = stablehlo.or %124, %126 : tensor<1xui32>
    %128 = stablehlo.xor %122, %127 : tensor<1xui32>
    %129 = stablehlo.add %122, %128 : tensor<1xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<1xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<1xui32>
    %134 = stablehlo.or %131, %133 : tensor<1xui32>
    %135 = stablehlo.xor %129, %134 : tensor<1xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %137 = stablehlo.add %129, %136 : tensor<1xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %139 = stablehlo.add %135, %138 : tensor<1xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %141 = stablehlo.add %139, %140 : tensor<1xui32>
    %142 = stablehlo.add %137, %141 : tensor<1xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<1xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<1xui32>
    %147 = stablehlo.or %144, %146 : tensor<1xui32>
    %148 = stablehlo.xor %142, %147 : tensor<1xui32>
    %149 = stablehlo.add %142, %148 : tensor<1xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<1xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<1xui32>
    %154 = stablehlo.or %151, %153 : tensor<1xui32>
    %155 = stablehlo.xor %149, %154 : tensor<1xui32>
    %156 = stablehlo.add %149, %155 : tensor<1xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<1xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<1xui32>
    %161 = stablehlo.or %158, %160 : tensor<1xui32>
    %162 = stablehlo.xor %156, %161 : tensor<1xui32>
    %163 = stablehlo.add %156, %162 : tensor<1xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<1xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<1xui32>
    %168 = stablehlo.or %165, %167 : tensor<1xui32>
    %169 = stablehlo.xor %163, %168 : tensor<1xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %171 = stablehlo.add %163, %170 : tensor<1xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %173 = stablehlo.add %169, %172 : tensor<1xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %175 = stablehlo.add %173, %174 : tensor<1xui32>
    return %171, %175 : tensor<1xui32>, tensor<1xui32>
  }
  func.func private @_normal(%arg0: tensor<2xui32>) -> tensor<151936x1024xf32> {
    %0 = call @_normal_real(%arg0) : (tensor<2xui32>) -> tensor<151936x1024xf32>
    return %0 : tensor<151936x1024xf32>
  }
  func.func private @_normal_real(%arg0: tensor<2xui32>) -> tensor<151936x1024xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<151936x1024xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<151936x1024xf32>
    %cst_1 = stablehlo.constant dense<2.83297682> : tensor<151936x1024xf32>
    %cst_2 = stablehlo.constant dense<1.50140941> : tensor<151936x1024xf32>
    %cst_3 = stablehlo.constant dense<1.00167406> : tensor<151936x1024xf32>
    %cst_4 = stablehlo.constant dense<0.246640727> : tensor<151936x1024xf32>
    %cst_5 = stablehlo.constant dense<0.00943887047> : tensor<151936x1024xf32>
    %cst_6 = stablehlo.constant dense<-0.00417768164> : tensor<151936x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.0076224613> : tensor<151936x1024xf32>
    %cst_8 = stablehlo.constant dense<-0.00125372503> : tensor<151936x1024xf32>
    %cst_9 = stablehlo.constant dense<0.00573950773> : tensor<151936x1024xf32>
    %cst_10 = stablehlo.constant dense<2.1858087E-4> : tensor<151936x1024xf32>
    %cst_11 = stablehlo.constant dense<-0.00367342844> : tensor<151936x1024xf32>
    %cst_12 = stablehlo.constant dense<-4.39150654E-6> : tensor<151936x1024xf32>
    %cst_13 = stablehlo.constant dense<0.00134934322> : tensor<151936x1024xf32>
    %cst_14 = stablehlo.constant dense<-3.5233877E-6> : tensor<151936x1024xf32>
    %cst_15 = stablehlo.constant dense<1.00950558E-4> : tensor<151936x1024xf32>
    %cst_16 = stablehlo.constant dense<3.43273939E-7> : tensor<151936x1024xf32>
    %cst_17 = stablehlo.constant dense<-2.00214257E-4> : tensor<151936x1024xf32>
    %cst_18 = stablehlo.constant dense<2.81022636E-8> : tensor<151936x1024xf32>
    %cst_19 = stablehlo.constant dense<3.000000e+00> : tensor<151936x1024xf32>
    %cst_20 = stablehlo.constant dense<2.500000e+00> : tensor<151936x1024xf32>
    %cst_21 = stablehlo.constant dense<5.000000e+00> : tensor<151936x1024xf32>
    %cst_22 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<151936x1024xf32>
    %1 = stablehlo.negate %0 : tensor<151936x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<151936x1024xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<151936x1024xf32>
    %4 = stablehlo.negate %3 : tensor<151936x1024xf32>
    %5 = stablehlo.compare LT, %4, %cst_21 : (tensor<151936x1024xf32>, tensor<151936x1024xf32>) -> tensor<151936x1024xi1>
    %6 = stablehlo.subtract %4, %cst_20 : tensor<151936x1024xf32>
    %7 = stablehlo.sqrt %4 : tensor<151936x1024xf32>
    %8 = stablehlo.subtract %7, %cst_19 : tensor<151936x1024xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %10 = stablehlo.select %5, %cst_18, %cst_17 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %11 = stablehlo.select %5, %cst_16, %cst_15 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<151936x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<151936x1024xf32>
    %14 = stablehlo.select %5, %cst_14, %cst_13 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<151936x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<151936x1024xf32>
    %17 = stablehlo.select %5, %cst_12, %cst_11 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<151936x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<151936x1024xf32>
    %20 = stablehlo.select %5, %cst_10, %cst_9 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<151936x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<151936x1024xf32>
    %23 = stablehlo.select %5, %cst_8, %cst_7 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<151936x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<151936x1024xf32>
    %26 = stablehlo.select %5, %cst_6, %cst_5 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<151936x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<151936x1024xf32>
    %29 = stablehlo.select %5, %cst_4, %cst_3 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<151936x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<151936x1024xf32>
    %32 = stablehlo.select %5, %cst_2, %cst_1 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<151936x1024xf32>
    %34 = stablehlo.add %32, %33 : tensor<151936x1024xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<151936x1024xf32>
    %36 = stablehlo.abs %0 : tensor<151936x1024xf32>
    %37 = stablehlo.compare EQ, %36, %cst_0 : (tensor<151936x1024xf32>, tensor<151936x1024xf32>) -> tensor<151936x1024xi1>
    %38 = stablehlo.multiply %0, %cst : tensor<151936x1024xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<151936x1024xi1>, tensor<151936x1024xf32>
    %40 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<151936x1024xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<151936x1024xf32>
    return %41 : tensor<151936x1024xf32>
  }
  func.func private @_uniform(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<151936x1024xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %c_2 = stablehlo.constant dense<1> : tensor<ui64>
    %c_3 = stablehlo.constant dense<1024> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<151936x1024xui64>
    %7 = stablehlo.iota dim = 1 : tensor<151936x1024xui64>
    %8 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<151936x1024xui64>
    %9 = stablehlo.multiply %8, %6 : tensor<151936x1024xui64>
    %10 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<151936x1024xui64>
    %11 = stablehlo.multiply %10, %7 : tensor<151936x1024xui64>
    %12 = stablehlo.add %9, %11 : tensor<151936x1024xui64>
    %13 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<151936x1024xui64>
    %14 = stablehlo.shift_right_logical %12, %13 : tensor<151936x1024xui64>
    %15 = stablehlo.convert %12 : (tensor<151936x1024xui64>) -> tensor<151936x1024xui32>
    %16 = stablehlo.convert %14 : (tensor<151936x1024xui64>) -> tensor<151936x1024xui32>
    %17:2 = call @threefry2x32_0(%3, %5, %16, %15) : (tensor<ui32>, tensor<ui32>, tensor<151936x1024xui32>, tensor<151936x1024xui32>) -> (tensor<151936x1024xui32>, tensor<151936x1024xui32>)
    %18 = stablehlo.xor %17#0, %17#1 : tensor<151936x1024xui32>
    %19 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %20 = stablehlo.shift_right_logical %18, %19 : tensor<151936x1024xui32>
    %21 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %22 = stablehlo.or %20, %21 : tensor<151936x1024xui32>
    %23 = stablehlo.bitcast_convert %22 : (tensor<151936x1024xui32>) -> tensor<151936x1024xf32>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<151936x1024xf32>
    %25 = stablehlo.subtract %23, %24 : tensor<151936x1024xf32>
    %26 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<151936x1024xf32>
    %28 = stablehlo.multiply %25, %27 : tensor<151936x1024xf32>
    %29 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<151936x1024xf32>
    %30 = stablehlo.add %28, %29 : tensor<151936x1024xf32>
    %31 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<151936x1024xf32>
    %32 = stablehlo.maximum %31, %30 : tensor<151936x1024xf32>
    return %32 : tensor<151936x1024xf32>
  }
  func.func private @threefry2x32_0(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<151936x1024xui32>, %arg3: tensor<151936x1024xui32>) -> (tensor<151936x1024xui32>, tensor<151936x1024xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<151936x1024xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<151936x1024xui32>
    %6 = stablehlo.add %3, %5 : tensor<151936x1024xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<151936x1024xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<151936x1024xui32>
    %11 = stablehlo.or %8, %10 : tensor<151936x1024xui32>
    %12 = stablehlo.xor %6, %11 : tensor<151936x1024xui32>
    %13 = stablehlo.add %6, %12 : tensor<151936x1024xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<151936x1024xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<151936x1024xui32>
    %18 = stablehlo.or %15, %17 : tensor<151936x1024xui32>
    %19 = stablehlo.xor %13, %18 : tensor<151936x1024xui32>
    %20 = stablehlo.add %13, %19 : tensor<151936x1024xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<151936x1024xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<151936x1024xui32>
    %25 = stablehlo.or %22, %24 : tensor<151936x1024xui32>
    %26 = stablehlo.xor %20, %25 : tensor<151936x1024xui32>
    %27 = stablehlo.add %20, %26 : tensor<151936x1024xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<151936x1024xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<151936x1024xui32>
    %32 = stablehlo.or %29, %31 : tensor<151936x1024xui32>
    %33 = stablehlo.xor %27, %32 : tensor<151936x1024xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %35 = stablehlo.add %27, %34 : tensor<151936x1024xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %37 = stablehlo.add %33, %36 : tensor<151936x1024xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %39 = stablehlo.add %37, %38 : tensor<151936x1024xui32>
    %40 = stablehlo.add %35, %39 : tensor<151936x1024xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<151936x1024xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<151936x1024xui32>
    %45 = stablehlo.or %42, %44 : tensor<151936x1024xui32>
    %46 = stablehlo.xor %40, %45 : tensor<151936x1024xui32>
    %47 = stablehlo.add %40, %46 : tensor<151936x1024xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<151936x1024xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<151936x1024xui32>
    %52 = stablehlo.or %49, %51 : tensor<151936x1024xui32>
    %53 = stablehlo.xor %47, %52 : tensor<151936x1024xui32>
    %54 = stablehlo.add %47, %53 : tensor<151936x1024xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<151936x1024xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<151936x1024xui32>
    %59 = stablehlo.or %56, %58 : tensor<151936x1024xui32>
    %60 = stablehlo.xor %54, %59 : tensor<151936x1024xui32>
    %61 = stablehlo.add %54, %60 : tensor<151936x1024xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<151936x1024xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<151936x1024xui32>
    %66 = stablehlo.or %63, %65 : tensor<151936x1024xui32>
    %67 = stablehlo.xor %61, %66 : tensor<151936x1024xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %69 = stablehlo.add %61, %68 : tensor<151936x1024xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %71 = stablehlo.add %67, %70 : tensor<151936x1024xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %73 = stablehlo.add %71, %72 : tensor<151936x1024xui32>
    %74 = stablehlo.add %69, %73 : tensor<151936x1024xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<151936x1024xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<151936x1024xui32>
    %79 = stablehlo.or %76, %78 : tensor<151936x1024xui32>
    %80 = stablehlo.xor %74, %79 : tensor<151936x1024xui32>
    %81 = stablehlo.add %74, %80 : tensor<151936x1024xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<151936x1024xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<151936x1024xui32>
    %86 = stablehlo.or %83, %85 : tensor<151936x1024xui32>
    %87 = stablehlo.xor %81, %86 : tensor<151936x1024xui32>
    %88 = stablehlo.add %81, %87 : tensor<151936x1024xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<151936x1024xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<151936x1024xui32>
    %93 = stablehlo.or %90, %92 : tensor<151936x1024xui32>
    %94 = stablehlo.xor %88, %93 : tensor<151936x1024xui32>
    %95 = stablehlo.add %88, %94 : tensor<151936x1024xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<151936x1024xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<151936x1024xui32>
    %100 = stablehlo.or %97, %99 : tensor<151936x1024xui32>
    %101 = stablehlo.xor %95, %100 : tensor<151936x1024xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %103 = stablehlo.add %95, %102 : tensor<151936x1024xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %105 = stablehlo.add %101, %104 : tensor<151936x1024xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %107 = stablehlo.add %105, %106 : tensor<151936x1024xui32>
    %108 = stablehlo.add %103, %107 : tensor<151936x1024xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<151936x1024xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<151936x1024xui32>
    %113 = stablehlo.or %110, %112 : tensor<151936x1024xui32>
    %114 = stablehlo.xor %108, %113 : tensor<151936x1024xui32>
    %115 = stablehlo.add %108, %114 : tensor<151936x1024xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<151936x1024xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<151936x1024xui32>
    %120 = stablehlo.or %117, %119 : tensor<151936x1024xui32>
    %121 = stablehlo.xor %115, %120 : tensor<151936x1024xui32>
    %122 = stablehlo.add %115, %121 : tensor<151936x1024xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<151936x1024xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<151936x1024xui32>
    %127 = stablehlo.or %124, %126 : tensor<151936x1024xui32>
    %128 = stablehlo.xor %122, %127 : tensor<151936x1024xui32>
    %129 = stablehlo.add %122, %128 : tensor<151936x1024xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<151936x1024xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<151936x1024xui32>
    %134 = stablehlo.or %131, %133 : tensor<151936x1024xui32>
    %135 = stablehlo.xor %129, %134 : tensor<151936x1024xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %137 = stablehlo.add %129, %136 : tensor<151936x1024xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %139 = stablehlo.add %135, %138 : tensor<151936x1024xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %141 = stablehlo.add %139, %140 : tensor<151936x1024xui32>
    %142 = stablehlo.add %137, %141 : tensor<151936x1024xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<151936x1024xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<151936x1024xui32>
    %147 = stablehlo.or %144, %146 : tensor<151936x1024xui32>
    %148 = stablehlo.xor %142, %147 : tensor<151936x1024xui32>
    %149 = stablehlo.add %142, %148 : tensor<151936x1024xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<151936x1024xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<151936x1024xui32>
    %154 = stablehlo.or %151, %153 : tensor<151936x1024xui32>
    %155 = stablehlo.xor %149, %154 : tensor<151936x1024xui32>
    %156 = stablehlo.add %149, %155 : tensor<151936x1024xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<151936x1024xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<151936x1024xui32>
    %161 = stablehlo.or %158, %160 : tensor<151936x1024xui32>
    %162 = stablehlo.xor %156, %161 : tensor<151936x1024xui32>
    %163 = stablehlo.add %156, %162 : tensor<151936x1024xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<151936x1024xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<151936x1024xui32>
    %168 = stablehlo.or %165, %167 : tensor<151936x1024xui32>
    %169 = stablehlo.xor %163, %168 : tensor<151936x1024xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %171 = stablehlo.add %163, %170 : tensor<151936x1024xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %173 = stablehlo.add %169, %172 : tensor<151936x1024xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<151936x1024xui32>
    %175 = stablehlo.add %173, %174 : tensor<151936x1024xui32>
    return %171, %175 : tensor<151936x1024xui32>, tensor<151936x1024xui32>
  }
  func.func private @_threefry_split(%arg0: tensor<2xui32>) -> tensor<28x2xui32> {
    %c = stablehlo.constant dense<32> : tensor<ui64>
    %c_0 = stablehlo.constant dense<1> : tensor<ui64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %1 = stablehlo.reshape %0 : (tensor<1xui32>) -> tensor<ui32>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.iota dim = 0 : tensor<28xui64>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<28xui64>
    %6 = stablehlo.multiply %5, %4 : tensor<28xui64>
    %7 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<28xui64>
    %8 = stablehlo.shift_right_logical %6, %7 : tensor<28xui64>
    %9 = stablehlo.convert %6 : (tensor<28xui64>) -> tensor<28xui32>
    %10 = stablehlo.convert %8 : (tensor<28xui64>) -> tensor<28xui32>
    %11:2 = call @threefry2x32_1(%1, %3, %10, %9) : (tensor<ui32>, tensor<ui32>, tensor<28xui32>, tensor<28xui32>) -> (tensor<28xui32>, tensor<28xui32>)
    %12 = stablehlo.broadcast_in_dim %11#0, dims = [0] : (tensor<28xui32>) -> tensor<28x1xui32>
    %13 = stablehlo.broadcast_in_dim %11#1, dims = [0] : (tensor<28xui32>) -> tensor<28x1xui32>
    %14 = stablehlo.concatenate %12, %13, dim = 1 : (tensor<28x1xui32>, tensor<28x1xui32>) -> tensor<28x2xui32>
    return %14 : tensor<28x2xui32>
  }
  func.func private @threefry2x32_1(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<28xui32>, %arg3: tensor<28xui32>) -> (tensor<28xui32>, tensor<28xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<28xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<28xui32>
    %6 = stablehlo.add %3, %5 : tensor<28xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<28xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<28xui32>
    %11 = stablehlo.or %8, %10 : tensor<28xui32>
    %12 = stablehlo.xor %6, %11 : tensor<28xui32>
    %13 = stablehlo.add %6, %12 : tensor<28xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<28xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<28xui32>
    %18 = stablehlo.or %15, %17 : tensor<28xui32>
    %19 = stablehlo.xor %13, %18 : tensor<28xui32>
    %20 = stablehlo.add %13, %19 : tensor<28xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<28xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<28xui32>
    %25 = stablehlo.or %22, %24 : tensor<28xui32>
    %26 = stablehlo.xor %20, %25 : tensor<28xui32>
    %27 = stablehlo.add %20, %26 : tensor<28xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<28xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<28xui32>
    %32 = stablehlo.or %29, %31 : tensor<28xui32>
    %33 = stablehlo.xor %27, %32 : tensor<28xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %35 = stablehlo.add %27, %34 : tensor<28xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %37 = stablehlo.add %33, %36 : tensor<28xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %39 = stablehlo.add %37, %38 : tensor<28xui32>
    %40 = stablehlo.add %35, %39 : tensor<28xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<28xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<28xui32>
    %45 = stablehlo.or %42, %44 : tensor<28xui32>
    %46 = stablehlo.xor %40, %45 : tensor<28xui32>
    %47 = stablehlo.add %40, %46 : tensor<28xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<28xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<28xui32>
    %52 = stablehlo.or %49, %51 : tensor<28xui32>
    %53 = stablehlo.xor %47, %52 : tensor<28xui32>
    %54 = stablehlo.add %47, %53 : tensor<28xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<28xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<28xui32>
    %59 = stablehlo.or %56, %58 : tensor<28xui32>
    %60 = stablehlo.xor %54, %59 : tensor<28xui32>
    %61 = stablehlo.add %54, %60 : tensor<28xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<28xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<28xui32>
    %66 = stablehlo.or %63, %65 : tensor<28xui32>
    %67 = stablehlo.xor %61, %66 : tensor<28xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %69 = stablehlo.add %61, %68 : tensor<28xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %71 = stablehlo.add %67, %70 : tensor<28xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %73 = stablehlo.add %71, %72 : tensor<28xui32>
    %74 = stablehlo.add %69, %73 : tensor<28xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<28xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<28xui32>
    %79 = stablehlo.or %76, %78 : tensor<28xui32>
    %80 = stablehlo.xor %74, %79 : tensor<28xui32>
    %81 = stablehlo.add %74, %80 : tensor<28xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<28xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<28xui32>
    %86 = stablehlo.or %83, %85 : tensor<28xui32>
    %87 = stablehlo.xor %81, %86 : tensor<28xui32>
    %88 = stablehlo.add %81, %87 : tensor<28xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<28xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<28xui32>
    %93 = stablehlo.or %90, %92 : tensor<28xui32>
    %94 = stablehlo.xor %88, %93 : tensor<28xui32>
    %95 = stablehlo.add %88, %94 : tensor<28xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<28xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<28xui32>
    %100 = stablehlo.or %97, %99 : tensor<28xui32>
    %101 = stablehlo.xor %95, %100 : tensor<28xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %103 = stablehlo.add %95, %102 : tensor<28xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %105 = stablehlo.add %101, %104 : tensor<28xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %107 = stablehlo.add %105, %106 : tensor<28xui32>
    %108 = stablehlo.add %103, %107 : tensor<28xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<28xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<28xui32>
    %113 = stablehlo.or %110, %112 : tensor<28xui32>
    %114 = stablehlo.xor %108, %113 : tensor<28xui32>
    %115 = stablehlo.add %108, %114 : tensor<28xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<28xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<28xui32>
    %120 = stablehlo.or %117, %119 : tensor<28xui32>
    %121 = stablehlo.xor %115, %120 : tensor<28xui32>
    %122 = stablehlo.add %115, %121 : tensor<28xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<28xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<28xui32>
    %127 = stablehlo.or %124, %126 : tensor<28xui32>
    %128 = stablehlo.xor %122, %127 : tensor<28xui32>
    %129 = stablehlo.add %122, %128 : tensor<28xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<28xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<28xui32>
    %134 = stablehlo.or %131, %133 : tensor<28xui32>
    %135 = stablehlo.xor %129, %134 : tensor<28xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %137 = stablehlo.add %129, %136 : tensor<28xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %139 = stablehlo.add %135, %138 : tensor<28xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %141 = stablehlo.add %139, %140 : tensor<28xui32>
    %142 = stablehlo.add %137, %141 : tensor<28xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<28xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<28xui32>
    %147 = stablehlo.or %144, %146 : tensor<28xui32>
    %148 = stablehlo.xor %142, %147 : tensor<28xui32>
    %149 = stablehlo.add %142, %148 : tensor<28xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<28xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<28xui32>
    %154 = stablehlo.or %151, %153 : tensor<28xui32>
    %155 = stablehlo.xor %149, %154 : tensor<28xui32>
    %156 = stablehlo.add %149, %155 : tensor<28xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<28xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<28xui32>
    %161 = stablehlo.or %158, %160 : tensor<28xui32>
    %162 = stablehlo.xor %156, %161 : tensor<28xui32>
    %163 = stablehlo.add %156, %162 : tensor<28xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<28xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<28xui32>
    %168 = stablehlo.or %165, %167 : tensor<28xui32>
    %169 = stablehlo.xor %163, %168 : tensor<28xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %171 = stablehlo.add %163, %170 : tensor<28xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %173 = stablehlo.add %169, %172 : tensor<28xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<28xui32>
    %175 = stablehlo.add %173, %174 : tensor<28xui32>
    return %171, %175 : tensor<28xui32>, tensor<28xui32>
  }
  func.func private @dynamic_index_in_dim(%arg0: tensor<28xui32>, %arg1: tensor<i32>) -> tensor<ui32> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [1] : (tensor<28xui32>, tensor<i32>) -> tensor<1xui32>
    %1 = stablehlo.reshape %0 : (tensor<1xui32>) -> tensor<ui32>
    return %1 : tensor<ui32>
  }
  func.func private @dynamic_index_in_dim_2(%arg0: tensor<28x2xui32>, %arg1: tensor<i32>) -> tensor<2xui32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [1, 2] : (tensor<28x2xui32>, tensor<i32>, tensor<i32>) -> tensor<1x2xui32>
    %1 = stablehlo.reshape %0 : (tensor<1x2xui32>) -> tensor<2xui32>
    return %1 : tensor<2xui32>
  }
  func.func private @closed_call(%arg0: tensor<ui32>, %arg1: tensor<2xui32>, %arg2: tensor<ui32>, %arg3: tensor<2xui32>, %arg4: tensor<ui32>, %arg5: tensor<2xui32>) -> (tensor<1024x3072xf32>, tensor<1024x3072xf32>, tensor<3072x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x8x128xf32>, tensor<128xf32>, tensor<16x128x1024xf32>, tensor<1024x16x128xf32>, tensor<128xf32>, tensor<1024x8x128xf32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) {
    %c = stablehlo.constant dense<0> : tensor<ui32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %c_1 = stablehlo.constant dense<-2> : tensor<i32>
    %cst_2 = stablehlo.constant dense<0.879625678> : tensor<f32>
    %cst_3 = stablehlo.constant dense<4.8828125E-4> : tensor<f32>
    %cst_4 = stablehlo.constant dense<9.765625E-4> : tensor<f32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_6 = stablehlo.constant dense<1> : tensor<ui32>
    %0 = stablehlo.add %arg4, %c_6 : tensor<ui32>
    %1 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %2 = call @_threefry_fold_in(%arg5, %0) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %3 = stablehlo.add %0, %c_6 : tensor<ui32>
    %4 = call @_normal_3(%2) : (tensor<2xui32>) -> tensor<1024x16x128xf32>
    %5 = stablehlo.sqrt %cst_4 : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1024x16x128xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1024x16x128xf32>
    %8 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x16x128xf32>
    %9 = stablehlo.divide %7, %8 : tensor<1024x16x128xf32>
    %10 = call @_threefry_fold_in(%arg5, %3) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %11 = stablehlo.add %3, %c_6 : tensor<ui32>
    %12 = call @_normal_7(%10) : (tensor<2xui32>) -> tensor<1024x8x128xf32>
    %13 = stablehlo.sqrt %cst_4 : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1024x8x128xf32>
    %15 = stablehlo.multiply %12, %14 : tensor<1024x8x128xf32>
    %16 = call @_threefry_fold_in(%arg5, %11) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %17 = stablehlo.add %11, %c_6 : tensor<ui32>
    %18 = call @_normal_7(%16) : (tensor<2xui32>) -> tensor<1024x8x128xf32>
    %19 = stablehlo.sqrt %cst_4 : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1024x8x128xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<1024x8x128xf32>
    %22 = call @_threefry_fold_in(%arg5, %17) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %23 = stablehlo.add %17, %c_6 : tensor<ui32>
    %24 = call @_normal_11(%22) : (tensor<2xui32>) -> tensor<16x128x1024xf32>
    %25 = stablehlo.sqrt %cst_3 : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<16x128x1024xf32>
    %27 = stablehlo.multiply %24, %26 : tensor<16x128x1024xf32>
    %28 = stablehlo.add %23, %c_6 : tensor<ui32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %30 = stablehlo.add %28, %c_6 : tensor<ui32>
    %31 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %32 = stablehlo.add %30, %c_6 : tensor<ui32>
    %33 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %34 = call @_threefry_fold_in(%arg5, %32) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %35 = stablehlo.add %32, %c_6 : tensor<ui32>
    %36 = stablehlo.sqrt %cst_4 : tensor<f32>
    %37 = stablehlo.divide %36, %cst_2 : tensor<f32>
    %38 = call @_truncated_normal(%34, %c_1, %c_0) : (tensor<2xui32>, tensor<i32>, tensor<i32>) -> tensor<1024x3072xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1024x3072xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<1024x3072xf32>
    %41 = call @_threefry_fold_in(%arg5, %35) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %42 = stablehlo.add %35, %c_6 : tensor<ui32>
    %43 = stablehlo.sqrt %cst_4 : tensor<f32>
    %44 = stablehlo.divide %43, %cst_2 : tensor<f32>
    %45 = call @_truncated_normal(%41, %c_1, %c_0) : (tensor<2xui32>, tensor<i32>, tensor<i32>) -> tensor<1024x3072xf32>
    %46 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f32>) -> tensor<1024x3072xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1024x3072xf32>
    %48 = call @_threefry_fold_in(%arg1, %arg0) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %49 = stablehlo.add %arg0, %c_6 : tensor<ui32>
    %50 = call @_threefry_fold_in(%arg3, %arg2) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %51 = stablehlo.add %arg2, %c_6 : tensor<ui32>
    %52 = call @_threefry_fold_in(%arg5, %42) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %53 = stablehlo.add %42, %c_6 : tensor<ui32>
    %54 = call @_threefry_fold_in(%arg5, %53) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %55 = stablehlo.add %53, %c_6 : tensor<ui32>
    %56 = stablehlo.sqrt %cst : tensor<f32>
    %57 = stablehlo.divide %56, %cst_2 : tensor<f32>
    %58 = call @_truncated_normal_17(%54, %c_1, %c_0) : (tensor<2xui32>, tensor<i32>, tensor<i32>) -> tensor<3072x1024xf32>
    %59 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<f32>) -> tensor<3072x1024xf32>
    %60 = stablehlo.multiply %58, %59 : tensor<3072x1024xf32>
    return %40, %47, %60, %33, %1, %15, %31, %27, %9, %29, %21, %c, %48, %c, %50, %c, %52, %49, %51, %55 : tensor<1024x3072xf32>, tensor<1024x3072xf32>, tensor<3072x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x8x128xf32>, tensor<128xf32>, tensor<16x128x1024xf32>, tensor<1024x16x128xf32>, tensor<128xf32>, tensor<1024x8x128xf32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>
  }
  func.func private @_normal_3(%arg0: tensor<2xui32>) -> tensor<1024x16x128xf32> {
    %0 = call @_normal_real_4(%arg0) : (tensor<2xui32>) -> tensor<1024x16x128xf32>
    return %0 : tensor<1024x16x128xf32>
  }
  func.func private @_normal_real_4(%arg0: tensor<2xui32>) -> tensor<1024x16x128xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<1024x16x128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1024x16x128xf32>
    %cst_1 = stablehlo.constant dense<2.83297682> : tensor<1024x16x128xf32>
    %cst_2 = stablehlo.constant dense<1.50140941> : tensor<1024x16x128xf32>
    %cst_3 = stablehlo.constant dense<1.00167406> : tensor<1024x16x128xf32>
    %cst_4 = stablehlo.constant dense<0.246640727> : tensor<1024x16x128xf32>
    %cst_5 = stablehlo.constant dense<0.00943887047> : tensor<1024x16x128xf32>
    %cst_6 = stablehlo.constant dense<-0.00417768164> : tensor<1024x16x128xf32>
    %cst_7 = stablehlo.constant dense<-0.0076224613> : tensor<1024x16x128xf32>
    %cst_8 = stablehlo.constant dense<-0.00125372503> : tensor<1024x16x128xf32>
    %cst_9 = stablehlo.constant dense<0.00573950773> : tensor<1024x16x128xf32>
    %cst_10 = stablehlo.constant dense<2.1858087E-4> : tensor<1024x16x128xf32>
    %cst_11 = stablehlo.constant dense<-0.00367342844> : tensor<1024x16x128xf32>
    %cst_12 = stablehlo.constant dense<-4.39150654E-6> : tensor<1024x16x128xf32>
    %cst_13 = stablehlo.constant dense<0.00134934322> : tensor<1024x16x128xf32>
    %cst_14 = stablehlo.constant dense<-3.5233877E-6> : tensor<1024x16x128xf32>
    %cst_15 = stablehlo.constant dense<1.00950558E-4> : tensor<1024x16x128xf32>
    %cst_16 = stablehlo.constant dense<3.43273939E-7> : tensor<1024x16x128xf32>
    %cst_17 = stablehlo.constant dense<-2.00214257E-4> : tensor<1024x16x128xf32>
    %cst_18 = stablehlo.constant dense<2.81022636E-8> : tensor<1024x16x128xf32>
    %cst_19 = stablehlo.constant dense<3.000000e+00> : tensor<1024x16x128xf32>
    %cst_20 = stablehlo.constant dense<2.500000e+00> : tensor<1024x16x128xf32>
    %cst_21 = stablehlo.constant dense<5.000000e+00> : tensor<1024x16x128xf32>
    %cst_22 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_5(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<1024x16x128xf32>
    %1 = stablehlo.negate %0 : tensor<1024x16x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x16x128xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<1024x16x128xf32>
    %4 = stablehlo.negate %3 : tensor<1024x16x128xf32>
    %5 = stablehlo.compare LT, %4, %cst_21 : (tensor<1024x16x128xf32>, tensor<1024x16x128xf32>) -> tensor<1024x16x128xi1>
    %6 = stablehlo.subtract %4, %cst_20 : tensor<1024x16x128xf32>
    %7 = stablehlo.sqrt %4 : tensor<1024x16x128xf32>
    %8 = stablehlo.subtract %7, %cst_19 : tensor<1024x16x128xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %10 = stablehlo.select %5, %cst_18, %cst_17 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %11 = stablehlo.select %5, %cst_16, %cst_15 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<1024x16x128xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x16x128xf32>
    %14 = stablehlo.select %5, %cst_14, %cst_13 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<1024x16x128xf32>
    %16 = stablehlo.add %14, %15 : tensor<1024x16x128xf32>
    %17 = stablehlo.select %5, %cst_12, %cst_11 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<1024x16x128xf32>
    %19 = stablehlo.add %17, %18 : tensor<1024x16x128xf32>
    %20 = stablehlo.select %5, %cst_10, %cst_9 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<1024x16x128xf32>
    %22 = stablehlo.add %20, %21 : tensor<1024x16x128xf32>
    %23 = stablehlo.select %5, %cst_8, %cst_7 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<1024x16x128xf32>
    %25 = stablehlo.add %23, %24 : tensor<1024x16x128xf32>
    %26 = stablehlo.select %5, %cst_6, %cst_5 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<1024x16x128xf32>
    %28 = stablehlo.add %26, %27 : tensor<1024x16x128xf32>
    %29 = stablehlo.select %5, %cst_4, %cst_3 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<1024x16x128xf32>
    %31 = stablehlo.add %29, %30 : tensor<1024x16x128xf32>
    %32 = stablehlo.select %5, %cst_2, %cst_1 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<1024x16x128xf32>
    %34 = stablehlo.add %32, %33 : tensor<1024x16x128xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<1024x16x128xf32>
    %36 = stablehlo.abs %0 : tensor<1024x16x128xf32>
    %37 = stablehlo.compare EQ, %36, %cst_0 : (tensor<1024x16x128xf32>, tensor<1024x16x128xf32>) -> tensor<1024x16x128xi1>
    %38 = stablehlo.multiply %0, %cst : tensor<1024x16x128xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<1024x16x128xi1>, tensor<1024x16x128xf32>
    %40 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1024x16x128xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<1024x16x128xf32>
    return %41 : tensor<1024x16x128xf32>
  }
  func.func private @_uniform_5(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1024x16x128xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %c_2 = stablehlo.constant dense<1> : tensor<ui64>
    %c_3 = stablehlo.constant dense<128> : tensor<ui64>
    %c_4 = stablehlo.constant dense<2048> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<1024x16x128xui64>
    %7 = stablehlo.iota dim = 1 : tensor<1024x16x128xui64>
    %8 = stablehlo.iota dim = 2 : tensor<1024x16x128xui64>
    %9 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<1024x16x128xui64>
    %10 = stablehlo.multiply %9, %6 : tensor<1024x16x128xui64>
    %11 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<1024x16x128xui64>
    %12 = stablehlo.multiply %11, %7 : tensor<1024x16x128xui64>
    %13 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<1024x16x128xui64>
    %14 = stablehlo.multiply %13, %8 : tensor<1024x16x128xui64>
    %15 = stablehlo.add %10, %12 : tensor<1024x16x128xui64>
    %16 = stablehlo.add %15, %14 : tensor<1024x16x128xui64>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<1024x16x128xui64>
    %18 = stablehlo.shift_right_logical %16, %17 : tensor<1024x16x128xui64>
    %19 = stablehlo.convert %16 : (tensor<1024x16x128xui64>) -> tensor<1024x16x128xui32>
    %20 = stablehlo.convert %18 : (tensor<1024x16x128xui64>) -> tensor<1024x16x128xui32>
    %21:2 = call @threefry2x32_6(%3, %5, %20, %19) : (tensor<ui32>, tensor<ui32>, tensor<1024x16x128xui32>, tensor<1024x16x128xui32>) -> (tensor<1024x16x128xui32>, tensor<1024x16x128xui32>)
    %22 = stablehlo.xor %21#0, %21#1 : tensor<1024x16x128xui32>
    %23 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %24 = stablehlo.shift_right_logical %22, %23 : tensor<1024x16x128xui32>
    %25 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %26 = stablehlo.or %24, %25 : tensor<1024x16x128xui32>
    %27 = stablehlo.bitcast_convert %26 : (tensor<1024x16x128xui32>) -> tensor<1024x16x128xf32>
    %28 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x16x128xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<1024x16x128xf32>
    %30 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1024x16x128xf32>
    %32 = stablehlo.multiply %29, %31 : tensor<1024x16x128xf32>
    %33 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1024x16x128xf32>
    %34 = stablehlo.add %32, %33 : tensor<1024x16x128xf32>
    %35 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1024x16x128xf32>
    %36 = stablehlo.maximum %35, %34 : tensor<1024x16x128xf32>
    return %36 : tensor<1024x16x128xf32>
  }
  func.func private @threefry2x32_6(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<1024x16x128xui32>, %arg3: tensor<1024x16x128xui32>) -> (tensor<1024x16x128xui32>, tensor<1024x16x128xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x16x128xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<1024x16x128xui32>
    %6 = stablehlo.add %3, %5 : tensor<1024x16x128xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<1024x16x128xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<1024x16x128xui32>
    %11 = stablehlo.or %8, %10 : tensor<1024x16x128xui32>
    %12 = stablehlo.xor %6, %11 : tensor<1024x16x128xui32>
    %13 = stablehlo.add %6, %12 : tensor<1024x16x128xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<1024x16x128xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<1024x16x128xui32>
    %18 = stablehlo.or %15, %17 : tensor<1024x16x128xui32>
    %19 = stablehlo.xor %13, %18 : tensor<1024x16x128xui32>
    %20 = stablehlo.add %13, %19 : tensor<1024x16x128xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<1024x16x128xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<1024x16x128xui32>
    %25 = stablehlo.or %22, %24 : tensor<1024x16x128xui32>
    %26 = stablehlo.xor %20, %25 : tensor<1024x16x128xui32>
    %27 = stablehlo.add %20, %26 : tensor<1024x16x128xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<1024x16x128xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<1024x16x128xui32>
    %32 = stablehlo.or %29, %31 : tensor<1024x16x128xui32>
    %33 = stablehlo.xor %27, %32 : tensor<1024x16x128xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %35 = stablehlo.add %27, %34 : tensor<1024x16x128xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %37 = stablehlo.add %33, %36 : tensor<1024x16x128xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %39 = stablehlo.add %37, %38 : tensor<1024x16x128xui32>
    %40 = stablehlo.add %35, %39 : tensor<1024x16x128xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<1024x16x128xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<1024x16x128xui32>
    %45 = stablehlo.or %42, %44 : tensor<1024x16x128xui32>
    %46 = stablehlo.xor %40, %45 : tensor<1024x16x128xui32>
    %47 = stablehlo.add %40, %46 : tensor<1024x16x128xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<1024x16x128xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<1024x16x128xui32>
    %52 = stablehlo.or %49, %51 : tensor<1024x16x128xui32>
    %53 = stablehlo.xor %47, %52 : tensor<1024x16x128xui32>
    %54 = stablehlo.add %47, %53 : tensor<1024x16x128xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<1024x16x128xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<1024x16x128xui32>
    %59 = stablehlo.or %56, %58 : tensor<1024x16x128xui32>
    %60 = stablehlo.xor %54, %59 : tensor<1024x16x128xui32>
    %61 = stablehlo.add %54, %60 : tensor<1024x16x128xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<1024x16x128xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<1024x16x128xui32>
    %66 = stablehlo.or %63, %65 : tensor<1024x16x128xui32>
    %67 = stablehlo.xor %61, %66 : tensor<1024x16x128xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %69 = stablehlo.add %61, %68 : tensor<1024x16x128xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %71 = stablehlo.add %67, %70 : tensor<1024x16x128xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %73 = stablehlo.add %71, %72 : tensor<1024x16x128xui32>
    %74 = stablehlo.add %69, %73 : tensor<1024x16x128xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<1024x16x128xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<1024x16x128xui32>
    %79 = stablehlo.or %76, %78 : tensor<1024x16x128xui32>
    %80 = stablehlo.xor %74, %79 : tensor<1024x16x128xui32>
    %81 = stablehlo.add %74, %80 : tensor<1024x16x128xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<1024x16x128xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<1024x16x128xui32>
    %86 = stablehlo.or %83, %85 : tensor<1024x16x128xui32>
    %87 = stablehlo.xor %81, %86 : tensor<1024x16x128xui32>
    %88 = stablehlo.add %81, %87 : tensor<1024x16x128xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<1024x16x128xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<1024x16x128xui32>
    %93 = stablehlo.or %90, %92 : tensor<1024x16x128xui32>
    %94 = stablehlo.xor %88, %93 : tensor<1024x16x128xui32>
    %95 = stablehlo.add %88, %94 : tensor<1024x16x128xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<1024x16x128xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<1024x16x128xui32>
    %100 = stablehlo.or %97, %99 : tensor<1024x16x128xui32>
    %101 = stablehlo.xor %95, %100 : tensor<1024x16x128xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %103 = stablehlo.add %95, %102 : tensor<1024x16x128xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %105 = stablehlo.add %101, %104 : tensor<1024x16x128xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %107 = stablehlo.add %105, %106 : tensor<1024x16x128xui32>
    %108 = stablehlo.add %103, %107 : tensor<1024x16x128xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<1024x16x128xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<1024x16x128xui32>
    %113 = stablehlo.or %110, %112 : tensor<1024x16x128xui32>
    %114 = stablehlo.xor %108, %113 : tensor<1024x16x128xui32>
    %115 = stablehlo.add %108, %114 : tensor<1024x16x128xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<1024x16x128xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<1024x16x128xui32>
    %120 = stablehlo.or %117, %119 : tensor<1024x16x128xui32>
    %121 = stablehlo.xor %115, %120 : tensor<1024x16x128xui32>
    %122 = stablehlo.add %115, %121 : tensor<1024x16x128xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<1024x16x128xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<1024x16x128xui32>
    %127 = stablehlo.or %124, %126 : tensor<1024x16x128xui32>
    %128 = stablehlo.xor %122, %127 : tensor<1024x16x128xui32>
    %129 = stablehlo.add %122, %128 : tensor<1024x16x128xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<1024x16x128xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<1024x16x128xui32>
    %134 = stablehlo.or %131, %133 : tensor<1024x16x128xui32>
    %135 = stablehlo.xor %129, %134 : tensor<1024x16x128xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %137 = stablehlo.add %129, %136 : tensor<1024x16x128xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %139 = stablehlo.add %135, %138 : tensor<1024x16x128xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %141 = stablehlo.add %139, %140 : tensor<1024x16x128xui32>
    %142 = stablehlo.add %137, %141 : tensor<1024x16x128xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<1024x16x128xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<1024x16x128xui32>
    %147 = stablehlo.or %144, %146 : tensor<1024x16x128xui32>
    %148 = stablehlo.xor %142, %147 : tensor<1024x16x128xui32>
    %149 = stablehlo.add %142, %148 : tensor<1024x16x128xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<1024x16x128xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<1024x16x128xui32>
    %154 = stablehlo.or %151, %153 : tensor<1024x16x128xui32>
    %155 = stablehlo.xor %149, %154 : tensor<1024x16x128xui32>
    %156 = stablehlo.add %149, %155 : tensor<1024x16x128xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<1024x16x128xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<1024x16x128xui32>
    %161 = stablehlo.or %158, %160 : tensor<1024x16x128xui32>
    %162 = stablehlo.xor %156, %161 : tensor<1024x16x128xui32>
    %163 = stablehlo.add %156, %162 : tensor<1024x16x128xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<1024x16x128xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<1024x16x128xui32>
    %168 = stablehlo.or %165, %167 : tensor<1024x16x128xui32>
    %169 = stablehlo.xor %163, %168 : tensor<1024x16x128xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %171 = stablehlo.add %163, %170 : tensor<1024x16x128xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %173 = stablehlo.add %169, %172 : tensor<1024x16x128xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024x16x128xui32>
    %175 = stablehlo.add %173, %174 : tensor<1024x16x128xui32>
    return %171, %175 : tensor<1024x16x128xui32>, tensor<1024x16x128xui32>
  }
  func.func private @_normal_7(%arg0: tensor<2xui32>) -> tensor<1024x8x128xf32> {
    %0 = call @_normal_real_8(%arg0) : (tensor<2xui32>) -> tensor<1024x8x128xf32>
    return %0 : tensor<1024x8x128xf32>
  }
  func.func private @_normal_real_8(%arg0: tensor<2xui32>) -> tensor<1024x8x128xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<1024x8x128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1024x8x128xf32>
    %cst_1 = stablehlo.constant dense<2.83297682> : tensor<1024x8x128xf32>
    %cst_2 = stablehlo.constant dense<1.50140941> : tensor<1024x8x128xf32>
    %cst_3 = stablehlo.constant dense<1.00167406> : tensor<1024x8x128xf32>
    %cst_4 = stablehlo.constant dense<0.246640727> : tensor<1024x8x128xf32>
    %cst_5 = stablehlo.constant dense<0.00943887047> : tensor<1024x8x128xf32>
    %cst_6 = stablehlo.constant dense<-0.00417768164> : tensor<1024x8x128xf32>
    %cst_7 = stablehlo.constant dense<-0.0076224613> : tensor<1024x8x128xf32>
    %cst_8 = stablehlo.constant dense<-0.00125372503> : tensor<1024x8x128xf32>
    %cst_9 = stablehlo.constant dense<0.00573950773> : tensor<1024x8x128xf32>
    %cst_10 = stablehlo.constant dense<2.1858087E-4> : tensor<1024x8x128xf32>
    %cst_11 = stablehlo.constant dense<-0.00367342844> : tensor<1024x8x128xf32>
    %cst_12 = stablehlo.constant dense<-4.39150654E-6> : tensor<1024x8x128xf32>
    %cst_13 = stablehlo.constant dense<0.00134934322> : tensor<1024x8x128xf32>
    %cst_14 = stablehlo.constant dense<-3.5233877E-6> : tensor<1024x8x128xf32>
    %cst_15 = stablehlo.constant dense<1.00950558E-4> : tensor<1024x8x128xf32>
    %cst_16 = stablehlo.constant dense<3.43273939E-7> : tensor<1024x8x128xf32>
    %cst_17 = stablehlo.constant dense<-2.00214257E-4> : tensor<1024x8x128xf32>
    %cst_18 = stablehlo.constant dense<2.81022636E-8> : tensor<1024x8x128xf32>
    %cst_19 = stablehlo.constant dense<3.000000e+00> : tensor<1024x8x128xf32>
    %cst_20 = stablehlo.constant dense<2.500000e+00> : tensor<1024x8x128xf32>
    %cst_21 = stablehlo.constant dense<5.000000e+00> : tensor<1024x8x128xf32>
    %cst_22 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_9(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<1024x8x128xf32>
    %1 = stablehlo.negate %0 : tensor<1024x8x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x8x128xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<1024x8x128xf32>
    %4 = stablehlo.negate %3 : tensor<1024x8x128xf32>
    %5 = stablehlo.compare LT, %4, %cst_21 : (tensor<1024x8x128xf32>, tensor<1024x8x128xf32>) -> tensor<1024x8x128xi1>
    %6 = stablehlo.subtract %4, %cst_20 : tensor<1024x8x128xf32>
    %7 = stablehlo.sqrt %4 : tensor<1024x8x128xf32>
    %8 = stablehlo.subtract %7, %cst_19 : tensor<1024x8x128xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %10 = stablehlo.select %5, %cst_18, %cst_17 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %11 = stablehlo.select %5, %cst_16, %cst_15 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<1024x8x128xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x8x128xf32>
    %14 = stablehlo.select %5, %cst_14, %cst_13 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<1024x8x128xf32>
    %16 = stablehlo.add %14, %15 : tensor<1024x8x128xf32>
    %17 = stablehlo.select %5, %cst_12, %cst_11 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<1024x8x128xf32>
    %19 = stablehlo.add %17, %18 : tensor<1024x8x128xf32>
    %20 = stablehlo.select %5, %cst_10, %cst_9 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<1024x8x128xf32>
    %22 = stablehlo.add %20, %21 : tensor<1024x8x128xf32>
    %23 = stablehlo.select %5, %cst_8, %cst_7 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<1024x8x128xf32>
    %25 = stablehlo.add %23, %24 : tensor<1024x8x128xf32>
    %26 = stablehlo.select %5, %cst_6, %cst_5 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<1024x8x128xf32>
    %28 = stablehlo.add %26, %27 : tensor<1024x8x128xf32>
    %29 = stablehlo.select %5, %cst_4, %cst_3 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<1024x8x128xf32>
    %31 = stablehlo.add %29, %30 : tensor<1024x8x128xf32>
    %32 = stablehlo.select %5, %cst_2, %cst_1 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<1024x8x128xf32>
    %34 = stablehlo.add %32, %33 : tensor<1024x8x128xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<1024x8x128xf32>
    %36 = stablehlo.abs %0 : tensor<1024x8x128xf32>
    %37 = stablehlo.compare EQ, %36, %cst_0 : (tensor<1024x8x128xf32>, tensor<1024x8x128xf32>) -> tensor<1024x8x128xi1>
    %38 = stablehlo.multiply %0, %cst : tensor<1024x8x128xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<1024x8x128xi1>, tensor<1024x8x128xf32>
    %40 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1024x8x128xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<1024x8x128xf32>
    return %41 : tensor<1024x8x128xf32>
  }
  func.func private @_uniform_9(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1024x8x128xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %c_2 = stablehlo.constant dense<1> : tensor<ui64>
    %c_3 = stablehlo.constant dense<128> : tensor<ui64>
    %c_4 = stablehlo.constant dense<1024> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<1024x8x128xui64>
    %7 = stablehlo.iota dim = 1 : tensor<1024x8x128xui64>
    %8 = stablehlo.iota dim = 2 : tensor<1024x8x128xui64>
    %9 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<1024x8x128xui64>
    %10 = stablehlo.multiply %9, %6 : tensor<1024x8x128xui64>
    %11 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<1024x8x128xui64>
    %12 = stablehlo.multiply %11, %7 : tensor<1024x8x128xui64>
    %13 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<1024x8x128xui64>
    %14 = stablehlo.multiply %13, %8 : tensor<1024x8x128xui64>
    %15 = stablehlo.add %10, %12 : tensor<1024x8x128xui64>
    %16 = stablehlo.add %15, %14 : tensor<1024x8x128xui64>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<1024x8x128xui64>
    %18 = stablehlo.shift_right_logical %16, %17 : tensor<1024x8x128xui64>
    %19 = stablehlo.convert %16 : (tensor<1024x8x128xui64>) -> tensor<1024x8x128xui32>
    %20 = stablehlo.convert %18 : (tensor<1024x8x128xui64>) -> tensor<1024x8x128xui32>
    %21:2 = call @threefry2x32_10(%3, %5, %20, %19) : (tensor<ui32>, tensor<ui32>, tensor<1024x8x128xui32>, tensor<1024x8x128xui32>) -> (tensor<1024x8x128xui32>, tensor<1024x8x128xui32>)
    %22 = stablehlo.xor %21#0, %21#1 : tensor<1024x8x128xui32>
    %23 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %24 = stablehlo.shift_right_logical %22, %23 : tensor<1024x8x128xui32>
    %25 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %26 = stablehlo.or %24, %25 : tensor<1024x8x128xui32>
    %27 = stablehlo.bitcast_convert %26 : (tensor<1024x8x128xui32>) -> tensor<1024x8x128xf32>
    %28 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x8x128xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<1024x8x128xf32>
    %30 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1024x8x128xf32>
    %32 = stablehlo.multiply %29, %31 : tensor<1024x8x128xf32>
    %33 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1024x8x128xf32>
    %34 = stablehlo.add %32, %33 : tensor<1024x8x128xf32>
    %35 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1024x8x128xf32>
    %36 = stablehlo.maximum %35, %34 : tensor<1024x8x128xf32>
    return %36 : tensor<1024x8x128xf32>
  }
  func.func private @threefry2x32_10(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<1024x8x128xui32>, %arg3: tensor<1024x8x128xui32>) -> (tensor<1024x8x128xui32>, tensor<1024x8x128xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x8x128xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<1024x8x128xui32>
    %6 = stablehlo.add %3, %5 : tensor<1024x8x128xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<1024x8x128xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<1024x8x128xui32>
    %11 = stablehlo.or %8, %10 : tensor<1024x8x128xui32>
    %12 = stablehlo.xor %6, %11 : tensor<1024x8x128xui32>
    %13 = stablehlo.add %6, %12 : tensor<1024x8x128xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<1024x8x128xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<1024x8x128xui32>
    %18 = stablehlo.or %15, %17 : tensor<1024x8x128xui32>
    %19 = stablehlo.xor %13, %18 : tensor<1024x8x128xui32>
    %20 = stablehlo.add %13, %19 : tensor<1024x8x128xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<1024x8x128xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<1024x8x128xui32>
    %25 = stablehlo.or %22, %24 : tensor<1024x8x128xui32>
    %26 = stablehlo.xor %20, %25 : tensor<1024x8x128xui32>
    %27 = stablehlo.add %20, %26 : tensor<1024x8x128xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<1024x8x128xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<1024x8x128xui32>
    %32 = stablehlo.or %29, %31 : tensor<1024x8x128xui32>
    %33 = stablehlo.xor %27, %32 : tensor<1024x8x128xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %35 = stablehlo.add %27, %34 : tensor<1024x8x128xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %37 = stablehlo.add %33, %36 : tensor<1024x8x128xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %39 = stablehlo.add %37, %38 : tensor<1024x8x128xui32>
    %40 = stablehlo.add %35, %39 : tensor<1024x8x128xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<1024x8x128xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<1024x8x128xui32>
    %45 = stablehlo.or %42, %44 : tensor<1024x8x128xui32>
    %46 = stablehlo.xor %40, %45 : tensor<1024x8x128xui32>
    %47 = stablehlo.add %40, %46 : tensor<1024x8x128xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<1024x8x128xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<1024x8x128xui32>
    %52 = stablehlo.or %49, %51 : tensor<1024x8x128xui32>
    %53 = stablehlo.xor %47, %52 : tensor<1024x8x128xui32>
    %54 = stablehlo.add %47, %53 : tensor<1024x8x128xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<1024x8x128xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<1024x8x128xui32>
    %59 = stablehlo.or %56, %58 : tensor<1024x8x128xui32>
    %60 = stablehlo.xor %54, %59 : tensor<1024x8x128xui32>
    %61 = stablehlo.add %54, %60 : tensor<1024x8x128xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<1024x8x128xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<1024x8x128xui32>
    %66 = stablehlo.or %63, %65 : tensor<1024x8x128xui32>
    %67 = stablehlo.xor %61, %66 : tensor<1024x8x128xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %69 = stablehlo.add %61, %68 : tensor<1024x8x128xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %71 = stablehlo.add %67, %70 : tensor<1024x8x128xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %73 = stablehlo.add %71, %72 : tensor<1024x8x128xui32>
    %74 = stablehlo.add %69, %73 : tensor<1024x8x128xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<1024x8x128xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<1024x8x128xui32>
    %79 = stablehlo.or %76, %78 : tensor<1024x8x128xui32>
    %80 = stablehlo.xor %74, %79 : tensor<1024x8x128xui32>
    %81 = stablehlo.add %74, %80 : tensor<1024x8x128xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<1024x8x128xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<1024x8x128xui32>
    %86 = stablehlo.or %83, %85 : tensor<1024x8x128xui32>
    %87 = stablehlo.xor %81, %86 : tensor<1024x8x128xui32>
    %88 = stablehlo.add %81, %87 : tensor<1024x8x128xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<1024x8x128xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<1024x8x128xui32>
    %93 = stablehlo.or %90, %92 : tensor<1024x8x128xui32>
    %94 = stablehlo.xor %88, %93 : tensor<1024x8x128xui32>
    %95 = stablehlo.add %88, %94 : tensor<1024x8x128xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<1024x8x128xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<1024x8x128xui32>
    %100 = stablehlo.or %97, %99 : tensor<1024x8x128xui32>
    %101 = stablehlo.xor %95, %100 : tensor<1024x8x128xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %103 = stablehlo.add %95, %102 : tensor<1024x8x128xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %105 = stablehlo.add %101, %104 : tensor<1024x8x128xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %107 = stablehlo.add %105, %106 : tensor<1024x8x128xui32>
    %108 = stablehlo.add %103, %107 : tensor<1024x8x128xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<1024x8x128xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<1024x8x128xui32>
    %113 = stablehlo.or %110, %112 : tensor<1024x8x128xui32>
    %114 = stablehlo.xor %108, %113 : tensor<1024x8x128xui32>
    %115 = stablehlo.add %108, %114 : tensor<1024x8x128xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<1024x8x128xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<1024x8x128xui32>
    %120 = stablehlo.or %117, %119 : tensor<1024x8x128xui32>
    %121 = stablehlo.xor %115, %120 : tensor<1024x8x128xui32>
    %122 = stablehlo.add %115, %121 : tensor<1024x8x128xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<1024x8x128xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<1024x8x128xui32>
    %127 = stablehlo.or %124, %126 : tensor<1024x8x128xui32>
    %128 = stablehlo.xor %122, %127 : tensor<1024x8x128xui32>
    %129 = stablehlo.add %122, %128 : tensor<1024x8x128xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<1024x8x128xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<1024x8x128xui32>
    %134 = stablehlo.or %131, %133 : tensor<1024x8x128xui32>
    %135 = stablehlo.xor %129, %134 : tensor<1024x8x128xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %137 = stablehlo.add %129, %136 : tensor<1024x8x128xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %139 = stablehlo.add %135, %138 : tensor<1024x8x128xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %141 = stablehlo.add %139, %140 : tensor<1024x8x128xui32>
    %142 = stablehlo.add %137, %141 : tensor<1024x8x128xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<1024x8x128xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<1024x8x128xui32>
    %147 = stablehlo.or %144, %146 : tensor<1024x8x128xui32>
    %148 = stablehlo.xor %142, %147 : tensor<1024x8x128xui32>
    %149 = stablehlo.add %142, %148 : tensor<1024x8x128xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<1024x8x128xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<1024x8x128xui32>
    %154 = stablehlo.or %151, %153 : tensor<1024x8x128xui32>
    %155 = stablehlo.xor %149, %154 : tensor<1024x8x128xui32>
    %156 = stablehlo.add %149, %155 : tensor<1024x8x128xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<1024x8x128xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<1024x8x128xui32>
    %161 = stablehlo.or %158, %160 : tensor<1024x8x128xui32>
    %162 = stablehlo.xor %156, %161 : tensor<1024x8x128xui32>
    %163 = stablehlo.add %156, %162 : tensor<1024x8x128xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<1024x8x128xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<1024x8x128xui32>
    %168 = stablehlo.or %165, %167 : tensor<1024x8x128xui32>
    %169 = stablehlo.xor %163, %168 : tensor<1024x8x128xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %171 = stablehlo.add %163, %170 : tensor<1024x8x128xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %173 = stablehlo.add %169, %172 : tensor<1024x8x128xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024x8x128xui32>
    %175 = stablehlo.add %173, %174 : tensor<1024x8x128xui32>
    return %171, %175 : tensor<1024x8x128xui32>, tensor<1024x8x128xui32>
  }
  func.func private @_normal_11(%arg0: tensor<2xui32>) -> tensor<16x128x1024xf32> {
    %0 = call @_normal_real_12(%arg0) : (tensor<2xui32>) -> tensor<16x128x1024xf32>
    return %0 : tensor<16x128x1024xf32>
  }
  func.func private @_normal_real_12(%arg0: tensor<2xui32>) -> tensor<16x128x1024xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<16x128x1024xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<16x128x1024xf32>
    %cst_1 = stablehlo.constant dense<2.83297682> : tensor<16x128x1024xf32>
    %cst_2 = stablehlo.constant dense<1.50140941> : tensor<16x128x1024xf32>
    %cst_3 = stablehlo.constant dense<1.00167406> : tensor<16x128x1024xf32>
    %cst_4 = stablehlo.constant dense<0.246640727> : tensor<16x128x1024xf32>
    %cst_5 = stablehlo.constant dense<0.00943887047> : tensor<16x128x1024xf32>
    %cst_6 = stablehlo.constant dense<-0.00417768164> : tensor<16x128x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.0076224613> : tensor<16x128x1024xf32>
    %cst_8 = stablehlo.constant dense<-0.00125372503> : tensor<16x128x1024xf32>
    %cst_9 = stablehlo.constant dense<0.00573950773> : tensor<16x128x1024xf32>
    %cst_10 = stablehlo.constant dense<2.1858087E-4> : tensor<16x128x1024xf32>
    %cst_11 = stablehlo.constant dense<-0.00367342844> : tensor<16x128x1024xf32>
    %cst_12 = stablehlo.constant dense<-4.39150654E-6> : tensor<16x128x1024xf32>
    %cst_13 = stablehlo.constant dense<0.00134934322> : tensor<16x128x1024xf32>
    %cst_14 = stablehlo.constant dense<-3.5233877E-6> : tensor<16x128x1024xf32>
    %cst_15 = stablehlo.constant dense<1.00950558E-4> : tensor<16x128x1024xf32>
    %cst_16 = stablehlo.constant dense<3.43273939E-7> : tensor<16x128x1024xf32>
    %cst_17 = stablehlo.constant dense<-2.00214257E-4> : tensor<16x128x1024xf32>
    %cst_18 = stablehlo.constant dense<2.81022636E-8> : tensor<16x128x1024xf32>
    %cst_19 = stablehlo.constant dense<3.000000e+00> : tensor<16x128x1024xf32>
    %cst_20 = stablehlo.constant dense<2.500000e+00> : tensor<16x128x1024xf32>
    %cst_21 = stablehlo.constant dense<5.000000e+00> : tensor<16x128x1024xf32>
    %cst_22 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_13(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<16x128x1024xf32>
    %1 = stablehlo.negate %0 : tensor<16x128x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x128x1024xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<16x128x1024xf32>
    %4 = stablehlo.negate %3 : tensor<16x128x1024xf32>
    %5 = stablehlo.compare LT, %4, %cst_21 : (tensor<16x128x1024xf32>, tensor<16x128x1024xf32>) -> tensor<16x128x1024xi1>
    %6 = stablehlo.subtract %4, %cst_20 : tensor<16x128x1024xf32>
    %7 = stablehlo.sqrt %4 : tensor<16x128x1024xf32>
    %8 = stablehlo.subtract %7, %cst_19 : tensor<16x128x1024xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %10 = stablehlo.select %5, %cst_18, %cst_17 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %11 = stablehlo.select %5, %cst_16, %cst_15 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<16x128x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<16x128x1024xf32>
    %14 = stablehlo.select %5, %cst_14, %cst_13 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<16x128x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x128x1024xf32>
    %17 = stablehlo.select %5, %cst_12, %cst_11 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<16x128x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x128x1024xf32>
    %20 = stablehlo.select %5, %cst_10, %cst_9 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<16x128x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x128x1024xf32>
    %23 = stablehlo.select %5, %cst_8, %cst_7 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<16x128x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x128x1024xf32>
    %26 = stablehlo.select %5, %cst_6, %cst_5 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<16x128x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x128x1024xf32>
    %29 = stablehlo.select %5, %cst_4, %cst_3 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<16x128x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x128x1024xf32>
    %32 = stablehlo.select %5, %cst_2, %cst_1 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<16x128x1024xf32>
    %34 = stablehlo.add %32, %33 : tensor<16x128x1024xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<16x128x1024xf32>
    %36 = stablehlo.abs %0 : tensor<16x128x1024xf32>
    %37 = stablehlo.compare EQ, %36, %cst_0 : (tensor<16x128x1024xf32>, tensor<16x128x1024xf32>) -> tensor<16x128x1024xi1>
    %38 = stablehlo.multiply %0, %cst : tensor<16x128x1024xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<16x128x1024xi1>, tensor<16x128x1024xf32>
    %40 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x128x1024xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<16x128x1024xf32>
    return %41 : tensor<16x128x1024xf32>
  }
  func.func private @_uniform_13(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<16x128x1024xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %c_2 = stablehlo.constant dense<1> : tensor<ui64>
    %c_3 = stablehlo.constant dense<1024> : tensor<ui64>
    %c_4 = stablehlo.constant dense<131072> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<16x128x1024xui64>
    %7 = stablehlo.iota dim = 1 : tensor<16x128x1024xui64>
    %8 = stablehlo.iota dim = 2 : tensor<16x128x1024xui64>
    %9 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<16x128x1024xui64>
    %10 = stablehlo.multiply %9, %6 : tensor<16x128x1024xui64>
    %11 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<16x128x1024xui64>
    %12 = stablehlo.multiply %11, %7 : tensor<16x128x1024xui64>
    %13 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<16x128x1024xui64>
    %14 = stablehlo.multiply %13, %8 : tensor<16x128x1024xui64>
    %15 = stablehlo.add %10, %12 : tensor<16x128x1024xui64>
    %16 = stablehlo.add %15, %14 : tensor<16x128x1024xui64>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<16x128x1024xui64>
    %18 = stablehlo.shift_right_logical %16, %17 : tensor<16x128x1024xui64>
    %19 = stablehlo.convert %16 : (tensor<16x128x1024xui64>) -> tensor<16x128x1024xui32>
    %20 = stablehlo.convert %18 : (tensor<16x128x1024xui64>) -> tensor<16x128x1024xui32>
    %21:2 = call @threefry2x32_14(%3, %5, %20, %19) : (tensor<ui32>, tensor<ui32>, tensor<16x128x1024xui32>, tensor<16x128x1024xui32>) -> (tensor<16x128x1024xui32>, tensor<16x128x1024xui32>)
    %22 = stablehlo.xor %21#0, %21#1 : tensor<16x128x1024xui32>
    %23 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %24 = stablehlo.shift_right_logical %22, %23 : tensor<16x128x1024xui32>
    %25 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %26 = stablehlo.or %24, %25 : tensor<16x128x1024xui32>
    %27 = stablehlo.bitcast_convert %26 : (tensor<16x128x1024xui32>) -> tensor<16x128x1024xf32>
    %28 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x1024xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<16x128x1024xf32>
    %30 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<16x128x1024xf32>
    %32 = stablehlo.multiply %29, %31 : tensor<16x128x1024xf32>
    %33 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<16x128x1024xf32>
    %34 = stablehlo.add %32, %33 : tensor<16x128x1024xf32>
    %35 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<16x128x1024xf32>
    %36 = stablehlo.maximum %35, %34 : tensor<16x128x1024xf32>
    return %36 : tensor<16x128x1024xf32>
  }
  func.func private @threefry2x32_14(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<16x128x1024xui32>, %arg3: tensor<16x128x1024xui32>) -> (tensor<16x128x1024xui32>, tensor<16x128x1024xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<16x128x1024xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<16x128x1024xui32>
    %6 = stablehlo.add %3, %5 : tensor<16x128x1024xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<16x128x1024xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<16x128x1024xui32>
    %11 = stablehlo.or %8, %10 : tensor<16x128x1024xui32>
    %12 = stablehlo.xor %6, %11 : tensor<16x128x1024xui32>
    %13 = stablehlo.add %6, %12 : tensor<16x128x1024xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<16x128x1024xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<16x128x1024xui32>
    %18 = stablehlo.or %15, %17 : tensor<16x128x1024xui32>
    %19 = stablehlo.xor %13, %18 : tensor<16x128x1024xui32>
    %20 = stablehlo.add %13, %19 : tensor<16x128x1024xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<16x128x1024xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<16x128x1024xui32>
    %25 = stablehlo.or %22, %24 : tensor<16x128x1024xui32>
    %26 = stablehlo.xor %20, %25 : tensor<16x128x1024xui32>
    %27 = stablehlo.add %20, %26 : tensor<16x128x1024xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<16x128x1024xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<16x128x1024xui32>
    %32 = stablehlo.or %29, %31 : tensor<16x128x1024xui32>
    %33 = stablehlo.xor %27, %32 : tensor<16x128x1024xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %35 = stablehlo.add %27, %34 : tensor<16x128x1024xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %37 = stablehlo.add %33, %36 : tensor<16x128x1024xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %39 = stablehlo.add %37, %38 : tensor<16x128x1024xui32>
    %40 = stablehlo.add %35, %39 : tensor<16x128x1024xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<16x128x1024xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<16x128x1024xui32>
    %45 = stablehlo.or %42, %44 : tensor<16x128x1024xui32>
    %46 = stablehlo.xor %40, %45 : tensor<16x128x1024xui32>
    %47 = stablehlo.add %40, %46 : tensor<16x128x1024xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<16x128x1024xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<16x128x1024xui32>
    %52 = stablehlo.or %49, %51 : tensor<16x128x1024xui32>
    %53 = stablehlo.xor %47, %52 : tensor<16x128x1024xui32>
    %54 = stablehlo.add %47, %53 : tensor<16x128x1024xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<16x128x1024xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<16x128x1024xui32>
    %59 = stablehlo.or %56, %58 : tensor<16x128x1024xui32>
    %60 = stablehlo.xor %54, %59 : tensor<16x128x1024xui32>
    %61 = stablehlo.add %54, %60 : tensor<16x128x1024xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<16x128x1024xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<16x128x1024xui32>
    %66 = stablehlo.or %63, %65 : tensor<16x128x1024xui32>
    %67 = stablehlo.xor %61, %66 : tensor<16x128x1024xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %69 = stablehlo.add %61, %68 : tensor<16x128x1024xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %71 = stablehlo.add %67, %70 : tensor<16x128x1024xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %73 = stablehlo.add %71, %72 : tensor<16x128x1024xui32>
    %74 = stablehlo.add %69, %73 : tensor<16x128x1024xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<16x128x1024xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<16x128x1024xui32>
    %79 = stablehlo.or %76, %78 : tensor<16x128x1024xui32>
    %80 = stablehlo.xor %74, %79 : tensor<16x128x1024xui32>
    %81 = stablehlo.add %74, %80 : tensor<16x128x1024xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<16x128x1024xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<16x128x1024xui32>
    %86 = stablehlo.or %83, %85 : tensor<16x128x1024xui32>
    %87 = stablehlo.xor %81, %86 : tensor<16x128x1024xui32>
    %88 = stablehlo.add %81, %87 : tensor<16x128x1024xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<16x128x1024xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<16x128x1024xui32>
    %93 = stablehlo.or %90, %92 : tensor<16x128x1024xui32>
    %94 = stablehlo.xor %88, %93 : tensor<16x128x1024xui32>
    %95 = stablehlo.add %88, %94 : tensor<16x128x1024xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<16x128x1024xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<16x128x1024xui32>
    %100 = stablehlo.or %97, %99 : tensor<16x128x1024xui32>
    %101 = stablehlo.xor %95, %100 : tensor<16x128x1024xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %103 = stablehlo.add %95, %102 : tensor<16x128x1024xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %105 = stablehlo.add %101, %104 : tensor<16x128x1024xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %107 = stablehlo.add %105, %106 : tensor<16x128x1024xui32>
    %108 = stablehlo.add %103, %107 : tensor<16x128x1024xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<16x128x1024xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<16x128x1024xui32>
    %113 = stablehlo.or %110, %112 : tensor<16x128x1024xui32>
    %114 = stablehlo.xor %108, %113 : tensor<16x128x1024xui32>
    %115 = stablehlo.add %108, %114 : tensor<16x128x1024xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<16x128x1024xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<16x128x1024xui32>
    %120 = stablehlo.or %117, %119 : tensor<16x128x1024xui32>
    %121 = stablehlo.xor %115, %120 : tensor<16x128x1024xui32>
    %122 = stablehlo.add %115, %121 : tensor<16x128x1024xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<16x128x1024xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<16x128x1024xui32>
    %127 = stablehlo.or %124, %126 : tensor<16x128x1024xui32>
    %128 = stablehlo.xor %122, %127 : tensor<16x128x1024xui32>
    %129 = stablehlo.add %122, %128 : tensor<16x128x1024xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<16x128x1024xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<16x128x1024xui32>
    %134 = stablehlo.or %131, %133 : tensor<16x128x1024xui32>
    %135 = stablehlo.xor %129, %134 : tensor<16x128x1024xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %137 = stablehlo.add %129, %136 : tensor<16x128x1024xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %139 = stablehlo.add %135, %138 : tensor<16x128x1024xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %141 = stablehlo.add %139, %140 : tensor<16x128x1024xui32>
    %142 = stablehlo.add %137, %141 : tensor<16x128x1024xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<16x128x1024xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<16x128x1024xui32>
    %147 = stablehlo.or %144, %146 : tensor<16x128x1024xui32>
    %148 = stablehlo.xor %142, %147 : tensor<16x128x1024xui32>
    %149 = stablehlo.add %142, %148 : tensor<16x128x1024xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<16x128x1024xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<16x128x1024xui32>
    %154 = stablehlo.or %151, %153 : tensor<16x128x1024xui32>
    %155 = stablehlo.xor %149, %154 : tensor<16x128x1024xui32>
    %156 = stablehlo.add %149, %155 : tensor<16x128x1024xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<16x128x1024xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<16x128x1024xui32>
    %161 = stablehlo.or %158, %160 : tensor<16x128x1024xui32>
    %162 = stablehlo.xor %156, %161 : tensor<16x128x1024xui32>
    %163 = stablehlo.add %156, %162 : tensor<16x128x1024xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<16x128x1024xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<16x128x1024xui32>
    %168 = stablehlo.or %165, %167 : tensor<16x128x1024xui32>
    %169 = stablehlo.xor %163, %168 : tensor<16x128x1024xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %171 = stablehlo.add %163, %170 : tensor<16x128x1024xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %173 = stablehlo.add %169, %172 : tensor<16x128x1024xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<16x128x1024xui32>
    %175 = stablehlo.add %173, %174 : tensor<16x128x1024xui32>
    return %171, %175 : tensor<16x128x1024xui32>, tensor<16x128x1024xui32>
  }
  func.func private @_truncated_normal(%arg0: tensor<2xui32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1024x3072xf32> {
    %c = stablehlo.constant dense<-1> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<2147483647> : tensor<i32>
    %c_3 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<0x7F800000> : tensor<1024x3072xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<1024x3072xf32>
    %cst_6 = stablehlo.constant dense<2.83297682> : tensor<1024x3072xf32>
    %cst_7 = stablehlo.constant dense<1.50140941> : tensor<1024x3072xf32>
    %cst_8 = stablehlo.constant dense<1.00167406> : tensor<1024x3072xf32>
    %cst_9 = stablehlo.constant dense<0.246640727> : tensor<1024x3072xf32>
    %cst_10 = stablehlo.constant dense<0.00943887047> : tensor<1024x3072xf32>
    %cst_11 = stablehlo.constant dense<-0.00417768164> : tensor<1024x3072xf32>
    %cst_12 = stablehlo.constant dense<-0.0076224613> : tensor<1024x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.00125372503> : tensor<1024x3072xf32>
    %cst_14 = stablehlo.constant dense<0.00573950773> : tensor<1024x3072xf32>
    %cst_15 = stablehlo.constant dense<2.1858087E-4> : tensor<1024x3072xf32>
    %cst_16 = stablehlo.constant dense<-0.00367342844> : tensor<1024x3072xf32>
    %cst_17 = stablehlo.constant dense<-4.39150654E-6> : tensor<1024x3072xf32>
    %cst_18 = stablehlo.constant dense<0.00134934322> : tensor<1024x3072xf32>
    %cst_19 = stablehlo.constant dense<-3.5233877E-6> : tensor<1024x3072xf32>
    %cst_20 = stablehlo.constant dense<1.00950558E-4> : tensor<1024x3072xf32>
    %cst_21 = stablehlo.constant dense<3.43273939E-7> : tensor<1024x3072xf32>
    %cst_22 = stablehlo.constant dense<-2.00214257E-4> : tensor<1024x3072xf32>
    %cst_23 = stablehlo.constant dense<2.81022636E-8> : tensor<1024x3072xf32>
    %cst_24 = stablehlo.constant dense<3.000000e+00> : tensor<1024x3072xf32>
    %cst_25 = stablehlo.constant dense<2.500000e+00> : tensor<1024x3072xf32>
    %cst_26 = stablehlo.constant dense<5.000000e+00> : tensor<1024x3072xf32>
    %cst_27 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_28 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %cst_29 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.divide %0, %cst_29 : tensor<f32>
    %3 = stablehlo.composite "chlo.erf" %2 {decomposition = @chlo.erf.impl_1, version = 1 : i32} : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.divide %1, %cst_29 : tensor<f32>
    %5 = stablehlo.composite "chlo.erf" %4 {decomposition = @chlo.erf.impl_2, version = 1 : i32} : (tensor<f32>) -> tensor<f32>
    %6 = call @_uniform_15(%arg0, %3, %5) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<1024x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1024x3072xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1024x3072xf32>
    %9 = stablehlo.log_plus_one %8 : tensor<1024x3072xf32>
    %10 = stablehlo.negate %9 : tensor<1024x3072xf32>
    %11 = stablehlo.compare LT, %10, %cst_26 : (tensor<1024x3072xf32>, tensor<1024x3072xf32>) -> tensor<1024x3072xi1>
    %12 = stablehlo.subtract %10, %cst_25 : tensor<1024x3072xf32>
    %13 = stablehlo.sqrt %10 : tensor<1024x3072xf32>
    %14 = stablehlo.subtract %13, %cst_24 : tensor<1024x3072xf32>
    %15 = stablehlo.select %11, %12, %14 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %16 = stablehlo.select %11, %cst_23, %cst_22 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %17 = stablehlo.select %11, %cst_21, %cst_20 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %18 = stablehlo.multiply %16, %15 : tensor<1024x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1024x3072xf32>
    %20 = stablehlo.select %11, %cst_19, %cst_18 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %21 = stablehlo.multiply %19, %15 : tensor<1024x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1024x3072xf32>
    %23 = stablehlo.select %11, %cst_17, %cst_16 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %24 = stablehlo.multiply %22, %15 : tensor<1024x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1024x3072xf32>
    %26 = stablehlo.select %11, %cst_15, %cst_14 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %27 = stablehlo.multiply %25, %15 : tensor<1024x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1024x3072xf32>
    %29 = stablehlo.select %11, %cst_13, %cst_12 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %30 = stablehlo.multiply %28, %15 : tensor<1024x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1024x3072xf32>
    %32 = stablehlo.select %11, %cst_11, %cst_10 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %33 = stablehlo.multiply %31, %15 : tensor<1024x3072xf32>
    %34 = stablehlo.add %32, %33 : tensor<1024x3072xf32>
    %35 = stablehlo.select %11, %cst_9, %cst_8 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %36 = stablehlo.multiply %34, %15 : tensor<1024x3072xf32>
    %37 = stablehlo.add %35, %36 : tensor<1024x3072xf32>
    %38 = stablehlo.select %11, %cst_7, %cst_6 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %39 = stablehlo.multiply %37, %15 : tensor<1024x3072xf32>
    %40 = stablehlo.add %38, %39 : tensor<1024x3072xf32>
    %41 = stablehlo.multiply %40, %6 : tensor<1024x3072xf32>
    %42 = stablehlo.abs %6 : tensor<1024x3072xf32>
    %43 = stablehlo.compare EQ, %42, %cst_5 : (tensor<1024x3072xf32>, tensor<1024x3072xf32>) -> tensor<1024x3072xi1>
    %44 = stablehlo.multiply %6, %cst_4 : tensor<1024x3072xf32>
    %45 = stablehlo.select %43, %44, %41 : tensor<1024x3072xi1>, tensor<1024x3072xf32>
    %46 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1024x3072xf32>
    %47 = stablehlo.multiply %46, %45 : tensor<1024x3072xf32>
    %48 = stablehlo.bitcast_convert %0 : (tensor<f32>) -> tensor<i32>
    %49 = stablehlo.bitcast_convert %cst_28 : (tensor<f32>) -> tensor<i32>
    %50 = stablehlo.compare NE, %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %51 = stablehlo.compare NE, %cst_28, %cst_28 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %52 = stablehlo.or %50, %51 : tensor<i1>
    %53 = stablehlo.bitcast_convert %cst : (tensor<f32>) -> tensor<i32>
    %54 = stablehlo.and %48, %c_2 : tensor<i32>
    %55 = stablehlo.and %49, %c_2 : tensor<i32>
    %56 = stablehlo.compare EQ, %0, %cst_28 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %57 = stablehlo.compare EQ, %54, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %58 = stablehlo.compare EQ, %55, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %59 = stablehlo.and %48, %c_3 : tensor<i32>
    %60 = stablehlo.and %49, %c_3 : tensor<i32>
    %61 = stablehlo.or %60, %c_0 : tensor<i32>
    %62 = stablehlo.compare NE, %59, %60 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %63 = stablehlo.compare GT, %54, %55 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %64 = stablehlo.or %63, %62 : tensor<i1>
    %65 = stablehlo.select %64, %c, %c_0 : tensor<i1>, tensor<i32>
    %66 = stablehlo.add %48, %65 : tensor<i32>
    %67 = stablehlo.select %58, %49, %61 : tensor<i1>, tensor<i32>
    %68 = stablehlo.select %57, %67, %66 : tensor<i1>, tensor<i32>
    %69 = stablehlo.select %56, %49, %68 : tensor<i1>, tensor<i32>
    %70 = stablehlo.select %52, %53, %69 : tensor<i1>, tensor<i32>
    %71 = stablehlo.bitcast_convert %70 : (tensor<i32>) -> tensor<f32>
    %72 = stablehlo.bitcast_convert %1 : (tensor<f32>) -> tensor<i32>
    %73 = stablehlo.bitcast_convert %cst_27 : (tensor<f32>) -> tensor<i32>
    %74 = stablehlo.compare NE, %1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %75 = stablehlo.compare NE, %cst_27, %cst_27 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %76 = stablehlo.or %74, %75 : tensor<i1>
    %77 = stablehlo.bitcast_convert %cst : (tensor<f32>) -> tensor<i32>
    %78 = stablehlo.and %72, %c_2 : tensor<i32>
    %79 = stablehlo.and %73, %c_2 : tensor<i32>
    %80 = stablehlo.compare EQ, %1, %cst_27 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %81 = stablehlo.compare EQ, %78, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %82 = stablehlo.compare EQ, %79, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %83 = stablehlo.and %72, %c_3 : tensor<i32>
    %84 = stablehlo.and %73, %c_3 : tensor<i32>
    %85 = stablehlo.or %84, %c_0 : tensor<i32>
    %86 = stablehlo.compare NE, %83, %84 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %87 = stablehlo.compare GT, %78, %79 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %88 = stablehlo.or %87, %86 : tensor<i1>
    %89 = stablehlo.select %88, %c, %c_0 : tensor<i1>, tensor<i32>
    %90 = stablehlo.add %72, %89 : tensor<i32>
    %91 = stablehlo.select %82, %73, %85 : tensor<i1>, tensor<i32>
    %92 = stablehlo.select %81, %91, %90 : tensor<i1>, tensor<i32>
    %93 = stablehlo.select %80, %73, %92 : tensor<i1>, tensor<i32>
    %94 = stablehlo.select %76, %77, %93 : tensor<i1>, tensor<i32>
    %95 = stablehlo.bitcast_convert %94 : (tensor<i32>) -> tensor<f32>
    %96 = call @clip(%47, %71, %95) : (tensor<1024x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<1024x3072xf32>
    return %96 : tensor<1024x3072xf32>
  }
  func.func private @chlo.erf.impl_2(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-0.0142647391> : tensor<f32>
    %cst_2 = stablehlo.constant dense<-0.00737332925> : tensor<f32>
    %cst_3 = stablehlo.constant dense<-0.00168282702> : tensor<f32>
    %cst_4 = stablehlo.constant dense<-2.13374049E-4> : tensor<f32>
    %cst_5 = stablehlo.constant dense<-1.45660715E-5> : tensor<f32>
    %cst_6 = stablehlo.constant dense<-0.0160960332> : tensor<f32>
    %cst_7 = stablehlo.constant dense<-2.954600e-03> : tensor<f32>
    %cst_8 = stablehlo.constant dense<-7.34990637E-4> : tensor<f32>
    %cst_9 = stablehlo.constant dense<-5.69250624E-5> : tensor<f32>
    %cst_10 = stablehlo.constant dense<-2.10102394E-6> : tensor<f32>
    %cst_11 = stablehlo.constant dense<2.77068146E-8> : tensor<f32>
    %cst_12 = stablehlo.constant dense<-2.72614237E-10> : tensor<f32>
    %cst_13 = stablehlo.constant dense<-4.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %0 = stablehlo.clamp %cst_13, %arg0, %cst_14 : tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.multiply %cst_12, %1 : tensor<f32>
    %3 = stablehlo.add %2, %cst_11 : tensor<f32>
    %4 = stablehlo.multiply %3, %1 : tensor<f32>
    %5 = stablehlo.add %4, %cst_10 : tensor<f32>
    %6 = stablehlo.multiply %5, %1 : tensor<f32>
    %7 = stablehlo.add %6, %cst_9 : tensor<f32>
    %8 = stablehlo.multiply %7, %1 : tensor<f32>
    %9 = stablehlo.add %8, %cst_8 : tensor<f32>
    %10 = stablehlo.multiply %9, %1 : tensor<f32>
    %11 = stablehlo.add %10, %cst_7 : tensor<f32>
    %12 = stablehlo.multiply %11, %1 : tensor<f32>
    %13 = stablehlo.add %12, %cst_6 : tensor<f32>
    %14 = stablehlo.multiply %cst_5, %1 : tensor<f32>
    %15 = stablehlo.add %14, %cst_4 : tensor<f32>
    %16 = stablehlo.multiply %15, %1 : tensor<f32>
    %17 = stablehlo.add %16, %cst_3 : tensor<f32>
    %18 = stablehlo.multiply %17, %1 : tensor<f32>
    %19 = stablehlo.add %18, %cst_2 : tensor<f32>
    %20 = stablehlo.multiply %19, %1 : tensor<f32>
    %21 = stablehlo.add %20, %cst_1 : tensor<f32>
    %22 = stablehlo.multiply %0, %13 : tensor<f32>
    %23 = stablehlo.divide %22, %21 : tensor<f32>
    %24 = stablehlo.clamp %cst_0, %23, %cst : tensor<f32>
    return %24 : tensor<f32>
  }
  func.func private @chlo.erf.impl_1(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-0.0142647391> : tensor<f32>
    %cst_2 = stablehlo.constant dense<-0.00737332925> : tensor<f32>
    %cst_3 = stablehlo.constant dense<-0.00168282702> : tensor<f32>
    %cst_4 = stablehlo.constant dense<-2.13374049E-4> : tensor<f32>
    %cst_5 = stablehlo.constant dense<-1.45660715E-5> : tensor<f32>
    %cst_6 = stablehlo.constant dense<-0.0160960332> : tensor<f32>
    %cst_7 = stablehlo.constant dense<-2.954600e-03> : tensor<f32>
    %cst_8 = stablehlo.constant dense<-7.34990637E-4> : tensor<f32>
    %cst_9 = stablehlo.constant dense<-5.69250624E-5> : tensor<f32>
    %cst_10 = stablehlo.constant dense<-2.10102394E-6> : tensor<f32>
    %cst_11 = stablehlo.constant dense<2.77068146E-8> : tensor<f32>
    %cst_12 = stablehlo.constant dense<-2.72614237E-10> : tensor<f32>
    %cst_13 = stablehlo.constant dense<-4.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %0 = stablehlo.clamp %cst_13, %arg0, %cst_14 : tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.multiply %cst_12, %1 : tensor<f32>
    %3 = stablehlo.add %2, %cst_11 : tensor<f32>
    %4 = stablehlo.multiply %3, %1 : tensor<f32>
    %5 = stablehlo.add %4, %cst_10 : tensor<f32>
    %6 = stablehlo.multiply %5, %1 : tensor<f32>
    %7 = stablehlo.add %6, %cst_9 : tensor<f32>
    %8 = stablehlo.multiply %7, %1 : tensor<f32>
    %9 = stablehlo.add %8, %cst_8 : tensor<f32>
    %10 = stablehlo.multiply %9, %1 : tensor<f32>
    %11 = stablehlo.add %10, %cst_7 : tensor<f32>
    %12 = stablehlo.multiply %11, %1 : tensor<f32>
    %13 = stablehlo.add %12, %cst_6 : tensor<f32>
    %14 = stablehlo.multiply %cst_5, %1 : tensor<f32>
    %15 = stablehlo.add %14, %cst_4 : tensor<f32>
    %16 = stablehlo.multiply %15, %1 : tensor<f32>
    %17 = stablehlo.add %16, %cst_3 : tensor<f32>
    %18 = stablehlo.multiply %17, %1 : tensor<f32>
    %19 = stablehlo.add %18, %cst_2 : tensor<f32>
    %20 = stablehlo.multiply %19, %1 : tensor<f32>
    %21 = stablehlo.add %20, %cst_1 : tensor<f32>
    %22 = stablehlo.multiply %0, %13 : tensor<f32>
    %23 = stablehlo.divide %22, %21 : tensor<f32>
    %24 = stablehlo.clamp %cst_0, %23, %cst : tensor<f32>
    return %24 : tensor<f32>
  }
  func.func private @_uniform_15(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1024x3072xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %c_2 = stablehlo.constant dense<1> : tensor<ui64>
    %c_3 = stablehlo.constant dense<3072> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<1024x3072xui64>
    %7 = stablehlo.iota dim = 1 : tensor<1024x3072xui64>
    %8 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<1024x3072xui64>
    %9 = stablehlo.multiply %8, %6 : tensor<1024x3072xui64>
    %10 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<1024x3072xui64>
    %11 = stablehlo.multiply %10, %7 : tensor<1024x3072xui64>
    %12 = stablehlo.add %9, %11 : tensor<1024x3072xui64>
    %13 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<1024x3072xui64>
    %14 = stablehlo.shift_right_logical %12, %13 : tensor<1024x3072xui64>
    %15 = stablehlo.convert %12 : (tensor<1024x3072xui64>) -> tensor<1024x3072xui32>
    %16 = stablehlo.convert %14 : (tensor<1024x3072xui64>) -> tensor<1024x3072xui32>
    %17:2 = call @threefry2x32_16(%3, %5, %16, %15) : (tensor<ui32>, tensor<ui32>, tensor<1024x3072xui32>, tensor<1024x3072xui32>) -> (tensor<1024x3072xui32>, tensor<1024x3072xui32>)
    %18 = stablehlo.xor %17#0, %17#1 : tensor<1024x3072xui32>
    %19 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %20 = stablehlo.shift_right_logical %18, %19 : tensor<1024x3072xui32>
    %21 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %22 = stablehlo.or %20, %21 : tensor<1024x3072xui32>
    %23 = stablehlo.bitcast_convert %22 : (tensor<1024x3072xui32>) -> tensor<1024x3072xf32>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x3072xf32>
    %25 = stablehlo.subtract %23, %24 : tensor<1024x3072xf32>
    %26 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1024x3072xf32>
    %28 = stablehlo.multiply %25, %27 : tensor<1024x3072xf32>
    %29 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1024x3072xf32>
    %30 = stablehlo.add %28, %29 : tensor<1024x3072xf32>
    %31 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1024x3072xf32>
    %32 = stablehlo.maximum %31, %30 : tensor<1024x3072xf32>
    return %32 : tensor<1024x3072xf32>
  }
  func.func private @threefry2x32_16(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<1024x3072xui32>, %arg3: tensor<1024x3072xui32>) -> (tensor<1024x3072xui32>, tensor<1024x3072xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x3072xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<1024x3072xui32>
    %6 = stablehlo.add %3, %5 : tensor<1024x3072xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<1024x3072xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<1024x3072xui32>
    %11 = stablehlo.or %8, %10 : tensor<1024x3072xui32>
    %12 = stablehlo.xor %6, %11 : tensor<1024x3072xui32>
    %13 = stablehlo.add %6, %12 : tensor<1024x3072xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<1024x3072xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<1024x3072xui32>
    %18 = stablehlo.or %15, %17 : tensor<1024x3072xui32>
    %19 = stablehlo.xor %13, %18 : tensor<1024x3072xui32>
    %20 = stablehlo.add %13, %19 : tensor<1024x3072xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<1024x3072xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<1024x3072xui32>
    %25 = stablehlo.or %22, %24 : tensor<1024x3072xui32>
    %26 = stablehlo.xor %20, %25 : tensor<1024x3072xui32>
    %27 = stablehlo.add %20, %26 : tensor<1024x3072xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<1024x3072xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<1024x3072xui32>
    %32 = stablehlo.or %29, %31 : tensor<1024x3072xui32>
    %33 = stablehlo.xor %27, %32 : tensor<1024x3072xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %35 = stablehlo.add %27, %34 : tensor<1024x3072xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %37 = stablehlo.add %33, %36 : tensor<1024x3072xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %39 = stablehlo.add %37, %38 : tensor<1024x3072xui32>
    %40 = stablehlo.add %35, %39 : tensor<1024x3072xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<1024x3072xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<1024x3072xui32>
    %45 = stablehlo.or %42, %44 : tensor<1024x3072xui32>
    %46 = stablehlo.xor %40, %45 : tensor<1024x3072xui32>
    %47 = stablehlo.add %40, %46 : tensor<1024x3072xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<1024x3072xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<1024x3072xui32>
    %52 = stablehlo.or %49, %51 : tensor<1024x3072xui32>
    %53 = stablehlo.xor %47, %52 : tensor<1024x3072xui32>
    %54 = stablehlo.add %47, %53 : tensor<1024x3072xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<1024x3072xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<1024x3072xui32>
    %59 = stablehlo.or %56, %58 : tensor<1024x3072xui32>
    %60 = stablehlo.xor %54, %59 : tensor<1024x3072xui32>
    %61 = stablehlo.add %54, %60 : tensor<1024x3072xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<1024x3072xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<1024x3072xui32>
    %66 = stablehlo.or %63, %65 : tensor<1024x3072xui32>
    %67 = stablehlo.xor %61, %66 : tensor<1024x3072xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %69 = stablehlo.add %61, %68 : tensor<1024x3072xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %71 = stablehlo.add %67, %70 : tensor<1024x3072xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %73 = stablehlo.add %71, %72 : tensor<1024x3072xui32>
    %74 = stablehlo.add %69, %73 : tensor<1024x3072xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<1024x3072xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<1024x3072xui32>
    %79 = stablehlo.or %76, %78 : tensor<1024x3072xui32>
    %80 = stablehlo.xor %74, %79 : tensor<1024x3072xui32>
    %81 = stablehlo.add %74, %80 : tensor<1024x3072xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<1024x3072xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<1024x3072xui32>
    %86 = stablehlo.or %83, %85 : tensor<1024x3072xui32>
    %87 = stablehlo.xor %81, %86 : tensor<1024x3072xui32>
    %88 = stablehlo.add %81, %87 : tensor<1024x3072xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<1024x3072xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<1024x3072xui32>
    %93 = stablehlo.or %90, %92 : tensor<1024x3072xui32>
    %94 = stablehlo.xor %88, %93 : tensor<1024x3072xui32>
    %95 = stablehlo.add %88, %94 : tensor<1024x3072xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<1024x3072xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<1024x3072xui32>
    %100 = stablehlo.or %97, %99 : tensor<1024x3072xui32>
    %101 = stablehlo.xor %95, %100 : tensor<1024x3072xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %103 = stablehlo.add %95, %102 : tensor<1024x3072xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %105 = stablehlo.add %101, %104 : tensor<1024x3072xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %107 = stablehlo.add %105, %106 : tensor<1024x3072xui32>
    %108 = stablehlo.add %103, %107 : tensor<1024x3072xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<1024x3072xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<1024x3072xui32>
    %113 = stablehlo.or %110, %112 : tensor<1024x3072xui32>
    %114 = stablehlo.xor %108, %113 : tensor<1024x3072xui32>
    %115 = stablehlo.add %108, %114 : tensor<1024x3072xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<1024x3072xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<1024x3072xui32>
    %120 = stablehlo.or %117, %119 : tensor<1024x3072xui32>
    %121 = stablehlo.xor %115, %120 : tensor<1024x3072xui32>
    %122 = stablehlo.add %115, %121 : tensor<1024x3072xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<1024x3072xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<1024x3072xui32>
    %127 = stablehlo.or %124, %126 : tensor<1024x3072xui32>
    %128 = stablehlo.xor %122, %127 : tensor<1024x3072xui32>
    %129 = stablehlo.add %122, %128 : tensor<1024x3072xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<1024x3072xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<1024x3072xui32>
    %134 = stablehlo.or %131, %133 : tensor<1024x3072xui32>
    %135 = stablehlo.xor %129, %134 : tensor<1024x3072xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %137 = stablehlo.add %129, %136 : tensor<1024x3072xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %139 = stablehlo.add %135, %138 : tensor<1024x3072xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %141 = stablehlo.add %139, %140 : tensor<1024x3072xui32>
    %142 = stablehlo.add %137, %141 : tensor<1024x3072xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<1024x3072xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<1024x3072xui32>
    %147 = stablehlo.or %144, %146 : tensor<1024x3072xui32>
    %148 = stablehlo.xor %142, %147 : tensor<1024x3072xui32>
    %149 = stablehlo.add %142, %148 : tensor<1024x3072xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<1024x3072xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<1024x3072xui32>
    %154 = stablehlo.or %151, %153 : tensor<1024x3072xui32>
    %155 = stablehlo.xor %149, %154 : tensor<1024x3072xui32>
    %156 = stablehlo.add %149, %155 : tensor<1024x3072xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<1024x3072xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<1024x3072xui32>
    %161 = stablehlo.or %158, %160 : tensor<1024x3072xui32>
    %162 = stablehlo.xor %156, %161 : tensor<1024x3072xui32>
    %163 = stablehlo.add %156, %162 : tensor<1024x3072xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<1024x3072xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<1024x3072xui32>
    %168 = stablehlo.or %165, %167 : tensor<1024x3072xui32>
    %169 = stablehlo.xor %163, %168 : tensor<1024x3072xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %171 = stablehlo.add %163, %170 : tensor<1024x3072xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %173 = stablehlo.add %169, %172 : tensor<1024x3072xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1024x3072xui32>
    %175 = stablehlo.add %173, %174 : tensor<1024x3072xui32>
    return %171, %175 : tensor<1024x3072xui32>, tensor<1024x3072xui32>
  }
  func.func private @clip(%arg0: tensor<1024x3072xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1024x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1024x3072xf32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<1024x3072xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1024x3072xf32>
    %3 = stablehlo.minimum %2, %1 : tensor<1024x3072xf32>
    return %3 : tensor<1024x3072xf32>
  }
  func.func private @_truncated_normal_17(%arg0: tensor<2xui32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<3072x1024xf32> {
    %c = stablehlo.constant dense<-1> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<2147483647> : tensor<i32>
    %c_3 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<0x7F800000> : tensor<3072x1024xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<3072x1024xf32>
    %cst_6 = stablehlo.constant dense<2.83297682> : tensor<3072x1024xf32>
    %cst_7 = stablehlo.constant dense<1.50140941> : tensor<3072x1024xf32>
    %cst_8 = stablehlo.constant dense<1.00167406> : tensor<3072x1024xf32>
    %cst_9 = stablehlo.constant dense<0.246640727> : tensor<3072x1024xf32>
    %cst_10 = stablehlo.constant dense<0.00943887047> : tensor<3072x1024xf32>
    %cst_11 = stablehlo.constant dense<-0.00417768164> : tensor<3072x1024xf32>
    %cst_12 = stablehlo.constant dense<-0.0076224613> : tensor<3072x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.00125372503> : tensor<3072x1024xf32>
    %cst_14 = stablehlo.constant dense<0.00573950773> : tensor<3072x1024xf32>
    %cst_15 = stablehlo.constant dense<2.1858087E-4> : tensor<3072x1024xf32>
    %cst_16 = stablehlo.constant dense<-0.00367342844> : tensor<3072x1024xf32>
    %cst_17 = stablehlo.constant dense<-4.39150654E-6> : tensor<3072x1024xf32>
    %cst_18 = stablehlo.constant dense<0.00134934322> : tensor<3072x1024xf32>
    %cst_19 = stablehlo.constant dense<-3.5233877E-6> : tensor<3072x1024xf32>
    %cst_20 = stablehlo.constant dense<1.00950558E-4> : tensor<3072x1024xf32>
    %cst_21 = stablehlo.constant dense<3.43273939E-7> : tensor<3072x1024xf32>
    %cst_22 = stablehlo.constant dense<-2.00214257E-4> : tensor<3072x1024xf32>
    %cst_23 = stablehlo.constant dense<2.81022636E-8> : tensor<3072x1024xf32>
    %cst_24 = stablehlo.constant dense<3.000000e+00> : tensor<3072x1024xf32>
    %cst_25 = stablehlo.constant dense<2.500000e+00> : tensor<3072x1024xf32>
    %cst_26 = stablehlo.constant dense<5.000000e+00> : tensor<3072x1024xf32>
    %cst_27 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_28 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %cst_29 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.divide %0, %cst_29 : tensor<f32>
    %3 = stablehlo.composite "chlo.erf" %2 {decomposition = @chlo.erf.impl, version = 1 : i32} : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.divide %1, %cst_29 : tensor<f32>
    %5 = stablehlo.composite "chlo.erf" %4 {decomposition = @chlo.erf.impl_0, version = 1 : i32} : (tensor<f32>) -> tensor<f32>
    %6 = call @_uniform_18(%arg0, %3, %5) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<3072x1024xf32>
    %7 = stablehlo.negate %6 : tensor<3072x1024xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<3072x1024xf32>
    %9 = stablehlo.log_plus_one %8 : tensor<3072x1024xf32>
    %10 = stablehlo.negate %9 : tensor<3072x1024xf32>
    %11 = stablehlo.compare LT, %10, %cst_26 : (tensor<3072x1024xf32>, tensor<3072x1024xf32>) -> tensor<3072x1024xi1>
    %12 = stablehlo.subtract %10, %cst_25 : tensor<3072x1024xf32>
    %13 = stablehlo.sqrt %10 : tensor<3072x1024xf32>
    %14 = stablehlo.subtract %13, %cst_24 : tensor<3072x1024xf32>
    %15 = stablehlo.select %11, %12, %14 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %16 = stablehlo.select %11, %cst_23, %cst_22 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %17 = stablehlo.select %11, %cst_21, %cst_20 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %18 = stablehlo.multiply %16, %15 : tensor<3072x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<3072x1024xf32>
    %20 = stablehlo.select %11, %cst_19, %cst_18 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %21 = stablehlo.multiply %19, %15 : tensor<3072x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<3072x1024xf32>
    %23 = stablehlo.select %11, %cst_17, %cst_16 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %24 = stablehlo.multiply %22, %15 : tensor<3072x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<3072x1024xf32>
    %26 = stablehlo.select %11, %cst_15, %cst_14 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %27 = stablehlo.multiply %25, %15 : tensor<3072x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<3072x1024xf32>
    %29 = stablehlo.select %11, %cst_13, %cst_12 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %30 = stablehlo.multiply %28, %15 : tensor<3072x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<3072x1024xf32>
    %32 = stablehlo.select %11, %cst_11, %cst_10 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %33 = stablehlo.multiply %31, %15 : tensor<3072x1024xf32>
    %34 = stablehlo.add %32, %33 : tensor<3072x1024xf32>
    %35 = stablehlo.select %11, %cst_9, %cst_8 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %36 = stablehlo.multiply %34, %15 : tensor<3072x1024xf32>
    %37 = stablehlo.add %35, %36 : tensor<3072x1024xf32>
    %38 = stablehlo.select %11, %cst_7, %cst_6 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %39 = stablehlo.multiply %37, %15 : tensor<3072x1024xf32>
    %40 = stablehlo.add %38, %39 : tensor<3072x1024xf32>
    %41 = stablehlo.multiply %40, %6 : tensor<3072x1024xf32>
    %42 = stablehlo.abs %6 : tensor<3072x1024xf32>
    %43 = stablehlo.compare EQ, %42, %cst_5 : (tensor<3072x1024xf32>, tensor<3072x1024xf32>) -> tensor<3072x1024xi1>
    %44 = stablehlo.multiply %6, %cst_4 : tensor<3072x1024xf32>
    %45 = stablehlo.select %43, %44, %41 : tensor<3072x1024xi1>, tensor<3072x1024xf32>
    %46 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<3072x1024xf32>
    %47 = stablehlo.multiply %46, %45 : tensor<3072x1024xf32>
    %48 = stablehlo.bitcast_convert %0 : (tensor<f32>) -> tensor<i32>
    %49 = stablehlo.bitcast_convert %cst_28 : (tensor<f32>) -> tensor<i32>
    %50 = stablehlo.compare NE, %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %51 = stablehlo.compare NE, %cst_28, %cst_28 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %52 = stablehlo.or %50, %51 : tensor<i1>
    %53 = stablehlo.bitcast_convert %cst : (tensor<f32>) -> tensor<i32>
    %54 = stablehlo.and %48, %c_2 : tensor<i32>
    %55 = stablehlo.and %49, %c_2 : tensor<i32>
    %56 = stablehlo.compare EQ, %0, %cst_28 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %57 = stablehlo.compare EQ, %54, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %58 = stablehlo.compare EQ, %55, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %59 = stablehlo.and %48, %c_3 : tensor<i32>
    %60 = stablehlo.and %49, %c_3 : tensor<i32>
    %61 = stablehlo.or %60, %c_0 : tensor<i32>
    %62 = stablehlo.compare NE, %59, %60 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %63 = stablehlo.compare GT, %54, %55 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %64 = stablehlo.or %63, %62 : tensor<i1>
    %65 = stablehlo.select %64, %c, %c_0 : tensor<i1>, tensor<i32>
    %66 = stablehlo.add %48, %65 : tensor<i32>
    %67 = stablehlo.select %58, %49, %61 : tensor<i1>, tensor<i32>
    %68 = stablehlo.select %57, %67, %66 : tensor<i1>, tensor<i32>
    %69 = stablehlo.select %56, %49, %68 : tensor<i1>, tensor<i32>
    %70 = stablehlo.select %52, %53, %69 : tensor<i1>, tensor<i32>
    %71 = stablehlo.bitcast_convert %70 : (tensor<i32>) -> tensor<f32>
    %72 = stablehlo.bitcast_convert %1 : (tensor<f32>) -> tensor<i32>
    %73 = stablehlo.bitcast_convert %cst_27 : (tensor<f32>) -> tensor<i32>
    %74 = stablehlo.compare NE, %1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %75 = stablehlo.compare NE, %cst_27, %cst_27 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %76 = stablehlo.or %74, %75 : tensor<i1>
    %77 = stablehlo.bitcast_convert %cst : (tensor<f32>) -> tensor<i32>
    %78 = stablehlo.and %72, %c_2 : tensor<i32>
    %79 = stablehlo.and %73, %c_2 : tensor<i32>
    %80 = stablehlo.compare EQ, %1, %cst_27 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %81 = stablehlo.compare EQ, %78, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %82 = stablehlo.compare EQ, %79, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %83 = stablehlo.and %72, %c_3 : tensor<i32>
    %84 = stablehlo.and %73, %c_3 : tensor<i32>
    %85 = stablehlo.or %84, %c_0 : tensor<i32>
    %86 = stablehlo.compare NE, %83, %84 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %87 = stablehlo.compare GT, %78, %79 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %88 = stablehlo.or %87, %86 : tensor<i1>
    %89 = stablehlo.select %88, %c, %c_0 : tensor<i1>, tensor<i32>
    %90 = stablehlo.add %72, %89 : tensor<i32>
    %91 = stablehlo.select %82, %73, %85 : tensor<i1>, tensor<i32>
    %92 = stablehlo.select %81, %91, %90 : tensor<i1>, tensor<i32>
    %93 = stablehlo.select %80, %73, %92 : tensor<i1>, tensor<i32>
    %94 = stablehlo.select %76, %77, %93 : tensor<i1>, tensor<i32>
    %95 = stablehlo.bitcast_convert %94 : (tensor<i32>) -> tensor<f32>
    %96 = call @clip_20(%47, %71, %95) : (tensor<3072x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<3072x1024xf32>
    return %96 : tensor<3072x1024xf32>
  }
  func.func private @chlo.erf.impl_0(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-0.0142647391> : tensor<f32>
    %cst_2 = stablehlo.constant dense<-0.00737332925> : tensor<f32>
    %cst_3 = stablehlo.constant dense<-0.00168282702> : tensor<f32>
    %cst_4 = stablehlo.constant dense<-2.13374049E-4> : tensor<f32>
    %cst_5 = stablehlo.constant dense<-1.45660715E-5> : tensor<f32>
    %cst_6 = stablehlo.constant dense<-0.0160960332> : tensor<f32>
    %cst_7 = stablehlo.constant dense<-2.954600e-03> : tensor<f32>
    %cst_8 = stablehlo.constant dense<-7.34990637E-4> : tensor<f32>
    %cst_9 = stablehlo.constant dense<-5.69250624E-5> : tensor<f32>
    %cst_10 = stablehlo.constant dense<-2.10102394E-6> : tensor<f32>
    %cst_11 = stablehlo.constant dense<2.77068146E-8> : tensor<f32>
    %cst_12 = stablehlo.constant dense<-2.72614237E-10> : tensor<f32>
    %cst_13 = stablehlo.constant dense<-4.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %0 = stablehlo.clamp %cst_13, %arg0, %cst_14 : tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.multiply %cst_12, %1 : tensor<f32>
    %3 = stablehlo.add %2, %cst_11 : tensor<f32>
    %4 = stablehlo.multiply %3, %1 : tensor<f32>
    %5 = stablehlo.add %4, %cst_10 : tensor<f32>
    %6 = stablehlo.multiply %5, %1 : tensor<f32>
    %7 = stablehlo.add %6, %cst_9 : tensor<f32>
    %8 = stablehlo.multiply %7, %1 : tensor<f32>
    %9 = stablehlo.add %8, %cst_8 : tensor<f32>
    %10 = stablehlo.multiply %9, %1 : tensor<f32>
    %11 = stablehlo.add %10, %cst_7 : tensor<f32>
    %12 = stablehlo.multiply %11, %1 : tensor<f32>
    %13 = stablehlo.add %12, %cst_6 : tensor<f32>
    %14 = stablehlo.multiply %cst_5, %1 : tensor<f32>
    %15 = stablehlo.add %14, %cst_4 : tensor<f32>
    %16 = stablehlo.multiply %15, %1 : tensor<f32>
    %17 = stablehlo.add %16, %cst_3 : tensor<f32>
    %18 = stablehlo.multiply %17, %1 : tensor<f32>
    %19 = stablehlo.add %18, %cst_2 : tensor<f32>
    %20 = stablehlo.multiply %19, %1 : tensor<f32>
    %21 = stablehlo.add %20, %cst_1 : tensor<f32>
    %22 = stablehlo.multiply %0, %13 : tensor<f32>
    %23 = stablehlo.divide %22, %21 : tensor<f32>
    %24 = stablehlo.clamp %cst_0, %23, %cst : tensor<f32>
    return %24 : tensor<f32>
  }
  func.func private @chlo.erf.impl(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-0.0142647391> : tensor<f32>
    %cst_2 = stablehlo.constant dense<-0.00737332925> : tensor<f32>
    %cst_3 = stablehlo.constant dense<-0.00168282702> : tensor<f32>
    %cst_4 = stablehlo.constant dense<-2.13374049E-4> : tensor<f32>
    %cst_5 = stablehlo.constant dense<-1.45660715E-5> : tensor<f32>
    %cst_6 = stablehlo.constant dense<-0.0160960332> : tensor<f32>
    %cst_7 = stablehlo.constant dense<-2.954600e-03> : tensor<f32>
    %cst_8 = stablehlo.constant dense<-7.34990637E-4> : tensor<f32>
    %cst_9 = stablehlo.constant dense<-5.69250624E-5> : tensor<f32>
    %cst_10 = stablehlo.constant dense<-2.10102394E-6> : tensor<f32>
    %cst_11 = stablehlo.constant dense<2.77068146E-8> : tensor<f32>
    %cst_12 = stablehlo.constant dense<-2.72614237E-10> : tensor<f32>
    %cst_13 = stablehlo.constant dense<-4.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %0 = stablehlo.clamp %cst_13, %arg0, %cst_14 : tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.multiply %cst_12, %1 : tensor<f32>
    %3 = stablehlo.add %2, %cst_11 : tensor<f32>
    %4 = stablehlo.multiply %3, %1 : tensor<f32>
    %5 = stablehlo.add %4, %cst_10 : tensor<f32>
    %6 = stablehlo.multiply %5, %1 : tensor<f32>
    %7 = stablehlo.add %6, %cst_9 : tensor<f32>
    %8 = stablehlo.multiply %7, %1 : tensor<f32>
    %9 = stablehlo.add %8, %cst_8 : tensor<f32>
    %10 = stablehlo.multiply %9, %1 : tensor<f32>
    %11 = stablehlo.add %10, %cst_7 : tensor<f32>
    %12 = stablehlo.multiply %11, %1 : tensor<f32>
    %13 = stablehlo.add %12, %cst_6 : tensor<f32>
    %14 = stablehlo.multiply %cst_5, %1 : tensor<f32>
    %15 = stablehlo.add %14, %cst_4 : tensor<f32>
    %16 = stablehlo.multiply %15, %1 : tensor<f32>
    %17 = stablehlo.add %16, %cst_3 : tensor<f32>
    %18 = stablehlo.multiply %17, %1 : tensor<f32>
    %19 = stablehlo.add %18, %cst_2 : tensor<f32>
    %20 = stablehlo.multiply %19, %1 : tensor<f32>
    %21 = stablehlo.add %20, %cst_1 : tensor<f32>
    %22 = stablehlo.multiply %0, %13 : tensor<f32>
    %23 = stablehlo.divide %22, %21 : tensor<f32>
    %24 = stablehlo.clamp %cst_0, %23, %cst : tensor<f32>
    return %24 : tensor<f32>
  }
  func.func private @_uniform_18(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3072x1024xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %c_2 = stablehlo.constant dense<1> : tensor<ui64>
    %c_3 = stablehlo.constant dense<1024> : tensor<ui64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<3072x1024xui64>
    %7 = stablehlo.iota dim = 1 : tensor<3072x1024xui64>
    %8 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<3072x1024xui64>
    %9 = stablehlo.multiply %8, %6 : tensor<3072x1024xui64>
    %10 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<3072x1024xui64>
    %11 = stablehlo.multiply %10, %7 : tensor<3072x1024xui64>
    %12 = stablehlo.add %9, %11 : tensor<3072x1024xui64>
    %13 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<3072x1024xui64>
    %14 = stablehlo.shift_right_logical %12, %13 : tensor<3072x1024xui64>
    %15 = stablehlo.convert %12 : (tensor<3072x1024xui64>) -> tensor<3072x1024xui32>
    %16 = stablehlo.convert %14 : (tensor<3072x1024xui64>) -> tensor<3072x1024xui32>
    %17:2 = call @threefry2x32_19(%3, %5, %16, %15) : (tensor<ui32>, tensor<ui32>, tensor<3072x1024xui32>, tensor<3072x1024xui32>) -> (tensor<3072x1024xui32>, tensor<3072x1024xui32>)
    %18 = stablehlo.xor %17#0, %17#1 : tensor<3072x1024xui32>
    %19 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %20 = stablehlo.shift_right_logical %18, %19 : tensor<3072x1024xui32>
    %21 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %22 = stablehlo.or %20, %21 : tensor<3072x1024xui32>
    %23 = stablehlo.bitcast_convert %22 : (tensor<3072x1024xui32>) -> tensor<3072x1024xf32>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3072x1024xf32>
    %25 = stablehlo.subtract %23, %24 : tensor<3072x1024xf32>
    %26 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<3072x1024xf32>
    %28 = stablehlo.multiply %25, %27 : tensor<3072x1024xf32>
    %29 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<3072x1024xf32>
    %30 = stablehlo.add %28, %29 : tensor<3072x1024xf32>
    %31 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<3072x1024xf32>
    %32 = stablehlo.maximum %31, %30 : tensor<3072x1024xf32>
    return %32 : tensor<3072x1024xf32>
  }
  func.func private @threefry2x32_19(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<3072x1024xui32>, %arg3: tensor<3072x1024xui32>) -> (tensor<3072x1024xui32>, tensor<3072x1024xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<3072x1024xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<3072x1024xui32>
    %6 = stablehlo.add %3, %5 : tensor<3072x1024xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<3072x1024xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<3072x1024xui32>
    %11 = stablehlo.or %8, %10 : tensor<3072x1024xui32>
    %12 = stablehlo.xor %6, %11 : tensor<3072x1024xui32>
    %13 = stablehlo.add %6, %12 : tensor<3072x1024xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<3072x1024xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<3072x1024xui32>
    %18 = stablehlo.or %15, %17 : tensor<3072x1024xui32>
    %19 = stablehlo.xor %13, %18 : tensor<3072x1024xui32>
    %20 = stablehlo.add %13, %19 : tensor<3072x1024xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<3072x1024xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<3072x1024xui32>
    %25 = stablehlo.or %22, %24 : tensor<3072x1024xui32>
    %26 = stablehlo.xor %20, %25 : tensor<3072x1024xui32>
    %27 = stablehlo.add %20, %26 : tensor<3072x1024xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<3072x1024xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<3072x1024xui32>
    %32 = stablehlo.or %29, %31 : tensor<3072x1024xui32>
    %33 = stablehlo.xor %27, %32 : tensor<3072x1024xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %35 = stablehlo.add %27, %34 : tensor<3072x1024xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %37 = stablehlo.add %33, %36 : tensor<3072x1024xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %39 = stablehlo.add %37, %38 : tensor<3072x1024xui32>
    %40 = stablehlo.add %35, %39 : tensor<3072x1024xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<3072x1024xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<3072x1024xui32>
    %45 = stablehlo.or %42, %44 : tensor<3072x1024xui32>
    %46 = stablehlo.xor %40, %45 : tensor<3072x1024xui32>
    %47 = stablehlo.add %40, %46 : tensor<3072x1024xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<3072x1024xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<3072x1024xui32>
    %52 = stablehlo.or %49, %51 : tensor<3072x1024xui32>
    %53 = stablehlo.xor %47, %52 : tensor<3072x1024xui32>
    %54 = stablehlo.add %47, %53 : tensor<3072x1024xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<3072x1024xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<3072x1024xui32>
    %59 = stablehlo.or %56, %58 : tensor<3072x1024xui32>
    %60 = stablehlo.xor %54, %59 : tensor<3072x1024xui32>
    %61 = stablehlo.add %54, %60 : tensor<3072x1024xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<3072x1024xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<3072x1024xui32>
    %66 = stablehlo.or %63, %65 : tensor<3072x1024xui32>
    %67 = stablehlo.xor %61, %66 : tensor<3072x1024xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %69 = stablehlo.add %61, %68 : tensor<3072x1024xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %71 = stablehlo.add %67, %70 : tensor<3072x1024xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %73 = stablehlo.add %71, %72 : tensor<3072x1024xui32>
    %74 = stablehlo.add %69, %73 : tensor<3072x1024xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<3072x1024xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<3072x1024xui32>
    %79 = stablehlo.or %76, %78 : tensor<3072x1024xui32>
    %80 = stablehlo.xor %74, %79 : tensor<3072x1024xui32>
    %81 = stablehlo.add %74, %80 : tensor<3072x1024xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<3072x1024xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<3072x1024xui32>
    %86 = stablehlo.or %83, %85 : tensor<3072x1024xui32>
    %87 = stablehlo.xor %81, %86 : tensor<3072x1024xui32>
    %88 = stablehlo.add %81, %87 : tensor<3072x1024xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<3072x1024xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<3072x1024xui32>
    %93 = stablehlo.or %90, %92 : tensor<3072x1024xui32>
    %94 = stablehlo.xor %88, %93 : tensor<3072x1024xui32>
    %95 = stablehlo.add %88, %94 : tensor<3072x1024xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<3072x1024xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<3072x1024xui32>
    %100 = stablehlo.or %97, %99 : tensor<3072x1024xui32>
    %101 = stablehlo.xor %95, %100 : tensor<3072x1024xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %103 = stablehlo.add %95, %102 : tensor<3072x1024xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %105 = stablehlo.add %101, %104 : tensor<3072x1024xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %107 = stablehlo.add %105, %106 : tensor<3072x1024xui32>
    %108 = stablehlo.add %103, %107 : tensor<3072x1024xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<3072x1024xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<3072x1024xui32>
    %113 = stablehlo.or %110, %112 : tensor<3072x1024xui32>
    %114 = stablehlo.xor %108, %113 : tensor<3072x1024xui32>
    %115 = stablehlo.add %108, %114 : tensor<3072x1024xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<3072x1024xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<3072x1024xui32>
    %120 = stablehlo.or %117, %119 : tensor<3072x1024xui32>
    %121 = stablehlo.xor %115, %120 : tensor<3072x1024xui32>
    %122 = stablehlo.add %115, %121 : tensor<3072x1024xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<3072x1024xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<3072x1024xui32>
    %127 = stablehlo.or %124, %126 : tensor<3072x1024xui32>
    %128 = stablehlo.xor %122, %127 : tensor<3072x1024xui32>
    %129 = stablehlo.add %122, %128 : tensor<3072x1024xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<3072x1024xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<3072x1024xui32>
    %134 = stablehlo.or %131, %133 : tensor<3072x1024xui32>
    %135 = stablehlo.xor %129, %134 : tensor<3072x1024xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %137 = stablehlo.add %129, %136 : tensor<3072x1024xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %139 = stablehlo.add %135, %138 : tensor<3072x1024xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %141 = stablehlo.add %139, %140 : tensor<3072x1024xui32>
    %142 = stablehlo.add %137, %141 : tensor<3072x1024xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<3072x1024xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<3072x1024xui32>
    %147 = stablehlo.or %144, %146 : tensor<3072x1024xui32>
    %148 = stablehlo.xor %142, %147 : tensor<3072x1024xui32>
    %149 = stablehlo.add %142, %148 : tensor<3072x1024xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<3072x1024xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<3072x1024xui32>
    %154 = stablehlo.or %151, %153 : tensor<3072x1024xui32>
    %155 = stablehlo.xor %149, %154 : tensor<3072x1024xui32>
    %156 = stablehlo.add %149, %155 : tensor<3072x1024xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<3072x1024xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<3072x1024xui32>
    %161 = stablehlo.or %158, %160 : tensor<3072x1024xui32>
    %162 = stablehlo.xor %156, %161 : tensor<3072x1024xui32>
    %163 = stablehlo.add %156, %162 : tensor<3072x1024xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<3072x1024xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<3072x1024xui32>
    %168 = stablehlo.or %165, %167 : tensor<3072x1024xui32>
    %169 = stablehlo.xor %163, %168 : tensor<3072x1024xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %171 = stablehlo.add %163, %170 : tensor<3072x1024xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %173 = stablehlo.add %169, %172 : tensor<3072x1024xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<3072x1024xui32>
    %175 = stablehlo.add %173, %174 : tensor<3072x1024xui32>
    return %171, %175 : tensor<3072x1024xui32>, tensor<3072x1024xui32>
  }
  func.func private @clip_20(%arg0: tensor<3072x1024xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3072x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<3072x1024xf32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<3072x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<3072x1024xf32>
    %3 = stablehlo.minimum %2, %1 : tensor<3072x1024xf32>
    return %3 : tensor<3072x1024xf32>
  }
  func.func private @dynamic_update_index_in_dim(%arg0: tensor<28x1024x3072xf32>, %arg1: tensor<1024x3072xf32>, %arg2: tensor<i32>) -> tensor<28x1024x3072xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<1024x3072xf32>) -> tensor<1x1024x3072xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c : (tensor<28x1024x3072xf32>, tensor<1x1024x3072xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x1024x3072xf32>
    return %1 : tensor<28x1024x3072xf32>
  }
  func.func private @dynamic_update_index_in_dim_21(%arg0: tensor<28x3072x1024xf32>, %arg1: tensor<3072x1024xf32>, %arg2: tensor<i32>) -> tensor<28x3072x1024xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<3072x1024xf32>) -> tensor<1x3072x1024xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c : (tensor<28x3072x1024xf32>, tensor<1x3072x1024xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x3072x1024xf32>
    return %1 : tensor<28x3072x1024xf32>
  }
  func.func private @dynamic_update_index_in_dim_22(%arg0: tensor<28x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<i32>) -> tensor<28x1024xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<28x1024xf32>, tensor<1x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<28x1024xf32>
    return %1 : tensor<28x1024xf32>
  }
  func.func private @dynamic_update_index_in_dim_23(%arg0: tensor<28x1024x8x128xf32>, %arg1: tensor<1024x8x128xf32>, %arg2: tensor<i32>) -> tensor<28x1024x8x128xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2, 3] : (tensor<1024x8x128xf32>) -> tensor<1x1024x8x128xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c, %c : (tensor<28x1024x8x128xf32>, tensor<1x1024x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x1024x8x128xf32>
    return %1 : tensor<28x1024x8x128xf32>
  }
  func.func private @dynamic_update_index_in_dim_24(%arg0: tensor<28x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<i32>) -> tensor<28x128xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<28x128xf32>, tensor<1x128xf32>, tensor<i32>, tensor<i32>) -> tensor<28x128xf32>
    return %1 : tensor<28x128xf32>
  }
  func.func private @dynamic_update_index_in_dim_25(%arg0: tensor<28x16x128x1024xf32>, %arg1: tensor<16x128x1024xf32>, %arg2: tensor<i32>) -> tensor<28x16x128x1024xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2, 3] : (tensor<16x128x1024xf32>) -> tensor<1x16x128x1024xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c, %c : (tensor<28x16x128x1024xf32>, tensor<1x16x128x1024xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x16x128x1024xf32>
    return %1 : tensor<28x16x128x1024xf32>
  }
  func.func private @dynamic_update_index_in_dim_26(%arg0: tensor<28x1024x16x128xf32>, %arg1: tensor<1024x16x128xf32>, %arg2: tensor<i32>) -> tensor<28x1024x16x128xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2, 3] : (tensor<1024x16x128xf32>) -> tensor<1x1024x16x128xf32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c, %c : (tensor<28x1024x16x128xf32>, tensor<1x1024x16x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x1024x16x128xf32>
    return %1 : tensor<28x1024x16x128xf32>
  }
  func.func private @dynamic_update_index_in_dim_27(%arg0: tensor<28xui32>, %arg1: tensor<ui32>, %arg2: tensor<i32>) -> tensor<28xui32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2 : (tensor<28xui32>, tensor<1xui32>, tensor<i32>) -> tensor<28xui32>
    return %1 : tensor<28xui32>
  }
  func.func private @dynamic_update_index_in_dim_28(%arg0: tensor<28x2xui32>, %arg1: tensor<2xui32>, %arg2: tensor<i32>) -> tensor<28x2xui32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xui32>) -> tensor<1x2xui32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<28x2xui32>, tensor<1x2xui32>, tensor<i32>, tensor<i32>) -> tensor<28x2xui32>
    return %1 : tensor<28x2xui32>
  }
}
