module @jit__prefill_jit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["diloco"=1, "data"=1, "stage"=1, "fsdp"=1, "fsdp_transpose"=1, "context"=1, "context_autoregressive"=1, "tensor"=1, "tensor_sequence"=1, "expert"=1, "autoregressive"=1]> {stablehlo.mesh = {axes = [{name = "diloco", size = 1 : i64}, {name = "data", size = 1 : i64}, {name = "stage", size = 1 : i64}, {name = "fsdp", size = 1 : i64}, {name = "fsdp_transpose", size = 1 : i64}, {name = "context", size = 1 : i64}, {name = "context_autoregressive", size = 1 : i64}, {name = "tensor", size = 1 : i64}, {name = "tensor_sequence", size = 1 : i64}, {name = "expert", size = 1 : i64}, {name = "autoregressive", size = 1 : i64}]}}
  func.func public @main(%arg0: tensor<16xi32>, %arg1: tensor<1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg2: tensor<1024x8x6144xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg3: tensor<1024x8x6144xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg4: tensor<6144x8x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg5: tensor<1024x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg6: tensor<1024x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg7: tensor<1024x8x8x128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, %arg8: tensor<128x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg9: tensor<16x8x128x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, %arg10: tensor<1024x8x16x128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, %arg11: tensor<128x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg12: tensor<1024x8x8x128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, %arg13: tensor<2048x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg14: tensor<i32>) -> (tensor<8x1xi32> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cache_ar_index']"}, tensor<8x1x10xi32> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cache_ar_segment_id']"}, tensor<8x1x16xi32> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cache_prefill_segment_id']"}, tensor<8x10x8x1x128xbf16> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cached_ar_key']"}, tensor<8x1xi32> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cached_ar_lengths']"}, tensor<8x10x8x1x128xbf16> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cached_ar_value']"}, tensor<8x16x8x1x128xbf16> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cached_prefill_key']"}, tensor<8x16x8x1x128xbf16> {jax.result_info = "result[0]['cache']['decoder']['layers']['self_attention']['KVCache_0']['cached_prefill_value']"}, tensor<1x1xi32> {jax.result_info = "result[0]['generated_tokens']"}, tensor<1x1x2048xf32> {jax.result_info = "result[0]['logits']"}, tensor<1x1xi32> {jax.result_info = "result[0]['next_pos']"}, tensor<1x1xf32> {jax.result_info = "result[0]['token_logp']"}, tensor<1x1xi32> {jax.result_info = "result[0]['tokens']"}, tensor<1x3xi32> {jax.result_info = "result[1][0]"}, tensor<1x1xf32> {jax.result_info = "result[1][1]"}) {
    %c = stablehlo.constant dense<1> : tensor<i8>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_3 = stablehlo.constant dense<8> : tensor<i32>
    %c_4 = stablehlo.constant dense<2048> : tensor<i32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %c_6 = stablehlo.constant dense<1> : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.add %c_7, %arg14 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16xi32>) -> tensor<1x16xi32>
    %2 = stablehlo.iota dim = 0 : tensor<16xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<16xi32>) -> tensor<1x16xi32>
    %4 = stablehlo.iota dim = 0 : tensor<16xi32>
    %5 = stablehlo.convert %0 : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %7 = stablehlo.compare LT, %4, %6, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %8 = stablehlo.convert %7 : (tensor<16xi1>) -> tensor<16xi32>
    %9 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %10 = stablehlo.multiply %8, %9 : tensor<16xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<16xi32>) -> tensor<1x16xi32>
    %12 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %13 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1x10xi32>
    %14 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<8x10x8x1x128xbf16>
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %16 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<8x10x8x1x128xbf16>
    %17 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %18 = stablehlo.compare LT, %1, %17, SIGNED : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi1>
    %19 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %20 = stablehlo.add %1, %19 : tensor<1x16xi32>
    %21 = stablehlo.select %18, %20, %1 : tensor<1x16xi1>, tensor<1x16xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1xi32>
    %23 = "stablehlo.gather"(%arg13, %22) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1024>}> : (tensor<2048x1024xbf16>, tensor<1x16x1xi32>) -> tensor<1x16x1024xbf16>
    %24 = stablehlo.transpose %arg2, dims = [1, 0, 2] : (tensor<1024x8x6144xbf16>) -> tensor<8x1024x6144xbf16>
    %25 = stablehlo.transpose %arg3, dims = [1, 0, 2] : (tensor<1024x8x6144xbf16>) -> tensor<8x1024x6144xbf16>
    %26 = stablehlo.transpose %arg4, dims = [1, 0, 2] : (tensor<6144x8x1024xbf16>) -> tensor<8x6144x1024xbf16>
    %27 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<1024x8xbf16>) -> tensor<8x1024xbf16>
    %28 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<1024x8xbf16>) -> tensor<8x1024xbf16>
    %29 = stablehlo.transpose %arg7, dims = [1, 0, 2, 3] : (tensor<1024x8x8x128xbf16>) -> tensor<8x1024x8x128xbf16>
    %30 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<128x8xbf16>) -> tensor<8x128xbf16>
    %31 = stablehlo.transpose %arg9, dims = [1, 0, 2, 3] : (tensor<16x8x128x1024xbf16>) -> tensor<8x16x128x1024xbf16>
    %32 = stablehlo.transpose %arg10, dims = [1, 0, 2, 3] : (tensor<1024x8x16x128xbf16>) -> tensor<8x1024x16x128xbf16>
    %33 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<128x8xbf16>) -> tensor<8x128xbf16>
    %34 = stablehlo.transpose %arg12, dims = [1, 0, 2, 3] : (tensor<1024x8x8x128xbf16>) -> tensor<8x1024x8x128xbf16>
    %35 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %36 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1x10xi32>
    %37 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1x16xi32>
    %38 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<8x10x8x1x128xbf16>
    %39 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %40 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<8x10x8x1x128xbf16>
    %41 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<8x16x8x1x128xbf16>
    %42 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<8x16x8x1x128xbf16>
    %43:28 = stablehlo.while(%iterArg = %24, %iterArg_8 = %25, %iterArg_9 = %26, %iterArg_10 = %27, %iterArg_11 = %28, %iterArg_12 = %29, %iterArg_13 = %30, %iterArg_14 = %31, %iterArg_15 = %32, %iterArg_16 = %33, %iterArg_17 = %34, %iterArg_18 = %12, %iterArg_19 = %13, %iterArg_20 = %14, %iterArg_21 = %15, %iterArg_22 = %16, %iterArg_23 = %3, %iterArg_24 = %11, %iterArg_25 = %c_7, %iterArg_26 = %23, %iterArg_27 = %35, %iterArg_28 = %36, %iterArg_29 = %37, %iterArg_30 = %38, %iterArg_31 = %39, %iterArg_32 = %40, %iterArg_33 = %41, %iterArg_34 = %42) : tensor<8x1024x6144xbf16>, tensor<8x1024x6144xbf16>, tensor<8x6144x1024xbf16>, tensor<8x1024xbf16>, tensor<8x1024xbf16>, tensor<8x1024x8x128xbf16>, tensor<8x128xbf16>, tensor<8x16x128x1024xbf16>, tensor<8x1024x16x128xbf16>, tensor<8x128xbf16>, tensor<8x1024x8x128xbf16>, tensor<8x1xi32>, tensor<8x1x10xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x1xi32>, tensor<8x10x8x1x128xbf16>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<i32>, tensor<1x16x1024xbf16>, tensor<8x1xi32>, tensor<8x1x10xi32>, tensor<8x1x16xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x1xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x16x8x1x128xbf16>, tensor<8x16x8x1x128xbf16>
    cond {
      %78 = stablehlo.compare LT, %iterArg_25, %c_3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %78 : tensor<i1>
    } do {
      %78 = func.call @dynamic_index_in_dim(%iterArg, %iterArg_25) : (tensor<8x1024x6144xbf16>, tensor<i32>) -> tensor<1024x6144xbf16>
      %79 = func.call @dynamic_index_in_dim(%iterArg_8, %iterArg_25) : (tensor<8x1024x6144xbf16>, tensor<i32>) -> tensor<1024x6144xbf16>
      %80 = func.call @dynamic_index_in_dim_0(%iterArg_9, %iterArg_25) : (tensor<8x6144x1024xbf16>, tensor<i32>) -> tensor<6144x1024xbf16>
      %81 = func.call @dynamic_index_in_dim_1(%iterArg_10, %iterArg_25) : (tensor<8x1024xbf16>, tensor<i32>) -> tensor<1024xbf16>
      %82 = func.call @dynamic_index_in_dim_1(%iterArg_11, %iterArg_25) : (tensor<8x1024xbf16>, tensor<i32>) -> tensor<1024xbf16>
      %83 = func.call @dynamic_index_in_dim_2(%iterArg_12, %iterArg_25) : (tensor<8x1024x8x128xbf16>, tensor<i32>) -> tensor<1024x8x128xbf16>
      %84 = func.call @dynamic_index_in_dim_3(%iterArg_13, %iterArg_25) : (tensor<8x128xbf16>, tensor<i32>) -> tensor<128xbf16>
      %85 = func.call @dynamic_index_in_dim_4(%iterArg_14, %iterArg_25) : (tensor<8x16x128x1024xbf16>, tensor<i32>) -> tensor<16x128x1024xbf16>
      %86 = func.call @dynamic_index_in_dim_5(%iterArg_15, %iterArg_25) : (tensor<8x1024x16x128xbf16>, tensor<i32>) -> tensor<1024x16x128xbf16>
      %87 = func.call @dynamic_index_in_dim_3(%iterArg_16, %iterArg_25) : (tensor<8x128xbf16>, tensor<i32>) -> tensor<128xbf16>
      %88 = func.call @dynamic_index_in_dim_2(%iterArg_17, %iterArg_25) : (tensor<8x1024x8x128xbf16>, tensor<i32>) -> tensor<1024x8x128xbf16>
      %89 = func.call @dynamic_index_in_dim_6(%iterArg_18, %iterArg_25) : (tensor<8x1xi32>, tensor<i32>) -> tensor<1xi32>
      %90 = func.call @dynamic_index_in_dim_7(%iterArg_19, %iterArg_25) : (tensor<8x1x10xi32>, tensor<i32>) -> tensor<1x10xi32>
      %91 = func.call @dynamic_index_in_dim_8(%iterArg_20, %iterArg_25) : (tensor<8x10x8x1x128xbf16>, tensor<i32>) -> tensor<10x8x1x128xbf16>
      %92 = func.call @dynamic_index_in_dim_6(%iterArg_21, %iterArg_25) : (tensor<8x1xi32>, tensor<i32>) -> tensor<1xi32>
      %93 = func.call @dynamic_index_in_dim_8(%iterArg_22, %iterArg_25) : (tensor<8x10x8x1x128xbf16>, tensor<i32>) -> tensor<10x8x1x128xbf16>
      %94:9 = func.call @closed_call(%iterArg_23, %iterArg_24, %iterArg_26, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93) : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16x1024xbf16>, tensor<1024x6144xbf16>, tensor<1024x6144xbf16>, tensor<6144x1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024x8x128xbf16>, tensor<128xbf16>, tensor<16x128x1024xbf16>, tensor<1024x16x128xbf16>, tensor<128xbf16>, tensor<1024x8x128xbf16>, tensor<1xi32>, tensor<1x10xi32>, tensor<10x8x1x128xbf16>, tensor<1xi32>, tensor<10x8x1x128xbf16>) -> (tensor<1x16x1024xbf16>, tensor<1xi32>, tensor<1x10xi32>, tensor<1x16xi32>, tensor<10x8x1x128xbf16>, tensor<1xi32>, tensor<10x8x1x128xbf16>, tensor<16x8x1x128xbf16>, tensor<16x8x1x128xbf16>)
      %95 = func.call @dynamic_update_index_in_dim(%iterArg_27, %94#1, %iterArg_25) : (tensor<8x1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<8x1xi32>
      %96 = func.call @dynamic_update_index_in_dim_10(%iterArg_28, %94#2, %iterArg_25) : (tensor<8x1x10xi32>, tensor<1x10xi32>, tensor<i32>) -> tensor<8x1x10xi32>
      %97 = func.call @dynamic_update_index_in_dim_11(%iterArg_29, %94#3, %iterArg_25) : (tensor<8x1x16xi32>, tensor<1x16xi32>, tensor<i32>) -> tensor<8x1x16xi32>
      %98 = func.call @dynamic_update_index_in_dim_12(%iterArg_30, %94#4, %iterArg_25) : (tensor<8x10x8x1x128xbf16>, tensor<10x8x1x128xbf16>, tensor<i32>) -> tensor<8x10x8x1x128xbf16>
      %99 = func.call @dynamic_update_index_in_dim(%iterArg_31, %94#5, %iterArg_25) : (tensor<8x1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<8x1xi32>
      %100 = func.call @dynamic_update_index_in_dim_12(%iterArg_32, %94#6, %iterArg_25) : (tensor<8x10x8x1x128xbf16>, tensor<10x8x1x128xbf16>, tensor<i32>) -> tensor<8x10x8x1x128xbf16>
      %101 = func.call @dynamic_update_index_in_dim_13(%iterArg_33, %94#7, %iterArg_25) : (tensor<8x16x8x1x128xbf16>, tensor<16x8x1x128xbf16>, tensor<i32>) -> tensor<8x16x8x1x128xbf16>
      %102 = func.call @dynamic_update_index_in_dim_13(%iterArg_34, %94#8, %iterArg_25) : (tensor<8x16x8x1x128xbf16>, tensor<16x8x1x128xbf16>, tensor<i32>) -> tensor<8x16x8x1x128xbf16>
      %103 = stablehlo.add %iterArg_25, %c_6 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %iterArg_24, %103, %94#0, %95, %96, %97, %98, %99, %100, %101, %102 : tensor<8x1024x6144xbf16>, tensor<8x1024x6144xbf16>, tensor<8x6144x1024xbf16>, tensor<8x1024xbf16>, tensor<8x1024xbf16>, tensor<8x1024x8x128xbf16>, tensor<8x128xbf16>, tensor<8x16x128x1024xbf16>, tensor<8x1024x16x128xbf16>, tensor<8x128xbf16>, tensor<8x1024x8x128xbf16>, tensor<8x1xi32>, tensor<8x1x10xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x1xi32>, tensor<8x10x8x1x128xbf16>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<i32>, tensor<1x16x1024xbf16>, tensor<8x1xi32>, tensor<8x1x10xi32>, tensor<8x1x16xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x1xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x16x8x1x128xbf16>, tensor<8x16x8x1x128xbf16>
    }
    %44 = stablehlo.convert %43#19 : (tensor<1x16x1024xbf16>) -> tensor<1x16x1024xf32>
    %45 = stablehlo.multiply %44, %44 : tensor<1x16x1024xf32>
    %46 = stablehlo.reduce(%45 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x16x1024xf32>, tensor<f32>) -> tensor<1x16xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %48 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %49 = stablehlo.divide %47, %48 : tensor<1x16x1xf32>
    %50 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x16x1xf32>
    %52 = stablehlo.rsqrt %51 : tensor<1x16x1xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1024xf32>
    %54 = stablehlo.multiply %44, %53 : tensor<1x16x1024xf32>
    %55 = stablehlo.convert %54 : (tensor<1x16x1024xf32>) -> tensor<1x16x1024xbf16>
    %56 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<1024xbf16>
    %57 = stablehlo.add %arg1, %56 : tensor<1024xbf16>
    %58 = stablehlo.dot_general %57, %55, batching_dims = [0] x [2], contracting_dims = [] x [] : (tensor<1024xbf16>, tensor<1x16x1024xbf16>) -> tensor<1024x1x16xbf16>
    %59 = stablehlo.transpose %58, dims = [1, 2, 0] : (tensor<1024x1x16xbf16>) -> tensor<1x16x1024xbf16>
    %60 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<2048x1024xbf16>) -> tensor<1024x2048xbf16>
    %61 = stablehlo.dot_general %59, %60, contracting_dims = [2] x [0] : (tensor<1x16x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x16x2048xbf16>
    %62 = stablehlo.convert %61 : (tensor<1x16x2048xbf16>) -> tensor<1x16x2048xf32>
    %63 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %64 = stablehlo.subtract %arg14, %c_6 : tensor<i32>
    %65 = stablehlo.compare LT, %64, %c_7, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %66 = stablehlo.convert %64 : tensor<i32>
    %67 = stablehlo.add %66, %c_0 : tensor<i32>
    %68 = stablehlo.select %65, %67, %64 : tensor<i1>, tensor<i32>
    %69 = stablehlo.dynamic_slice %62, %c_7, %68, %c_7, sizes = [1, 1, 2048] : (tensor<1x16x2048xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x2048xf32>
    %70 = sdy.sharding_constraint %69 <@mesh, [{}, {}, {}]> : tensor<1x1x2048xf32>
    %71 = call @argmax(%70) : (tensor<1x1x2048xf32>) -> tensor<1x1xi32>
    %72 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i8>) -> tensor<1x1xi8>
    %73 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %74 = stablehlo.convert %72 : (tensor<1x1xi8>) -> tensor<1x1xi32>
    %75 = stablehlo.concatenate %71, %74, %63, dim = 1 : (tensor<1x1xi32>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x3xi32>
    %76 = stablehlo.convert %0 : tensor<i32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    return %43#20, %43#21, %43#22, %43#23, %43#24, %43#25, %43#26, %43#27, %63, %70, %77, %73, %71, %75, %73 : tensor<8x1xi32>, tensor<8x1x10xi32>, tensor<8x1x16xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x1xi32>, tensor<8x10x8x1x128xbf16>, tensor<8x16x8x1x128xbf16>, tensor<8x16x8x1x128xbf16>, tensor<1x1xi32>, tensor<1x1x2048xf32>, tensor<1x1xi32>, tensor<1x1xf32>, tensor<1x1xi32>, tensor<1x3xi32>, tensor<1x1xf32>
  }
  func.func private @dynamic_index_in_dim(%arg0: tensor<8x1024x6144xbf16>, %arg1: tensor<i32>) -> tensor<1024x6144xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, sizes = [1, 1024, 6144] : (tensor<8x1024x6144xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1024x6144xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x1024x6144xbf16>) -> tensor<1024x6144xbf16>
    return %1 : tensor<1024x6144xbf16>
  }
  func.func private @dynamic_index_in_dim_0(%arg0: tensor<8x6144x1024xbf16>, %arg1: tensor<i32>) -> tensor<6144x1024xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, sizes = [1, 6144, 1024] : (tensor<8x6144x1024xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x6144x1024xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x6144x1024xbf16>) -> tensor<6144x1024xbf16>
    return %1 : tensor<6144x1024xbf16>
  }
  func.func private @dynamic_index_in_dim_1(%arg0: tensor<8x1024xbf16>, %arg1: tensor<i32>) -> tensor<1024xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [1, 1024] : (tensor<8x1024xbf16>, tensor<i32>, tensor<i32>) -> tensor<1x1024xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x1024xbf16>) -> tensor<1024xbf16>
    return %1 : tensor<1024xbf16>
  }
  func.func private @dynamic_index_in_dim_2(%arg0: tensor<8x1024x8x128xbf16>, %arg1: tensor<i32>) -> tensor<1024x8x128xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, %c, sizes = [1, 1024, 8, 128] : (tensor<8x1024x8x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1024x8x128xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x1024x8x128xbf16>) -> tensor<1024x8x128xbf16>
    return %1 : tensor<1024x8x128xbf16>
  }
  func.func private @dynamic_index_in_dim_3(%arg0: tensor<8x128xbf16>, %arg1: tensor<i32>) -> tensor<128xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [1, 128] : (tensor<8x128xbf16>, tensor<i32>, tensor<i32>) -> tensor<1x128xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x128xbf16>) -> tensor<128xbf16>
    return %1 : tensor<128xbf16>
  }
  func.func private @dynamic_index_in_dim_4(%arg0: tensor<8x16x128x1024xbf16>, %arg1: tensor<i32>) -> tensor<16x128x1024xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, %c, sizes = [1, 16, 128, 1024] : (tensor<8x16x128x1024xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x16x128x1024xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x16x128x1024xbf16>) -> tensor<16x128x1024xbf16>
    return %1 : tensor<16x128x1024xbf16>
  }
  func.func private @dynamic_index_in_dim_5(%arg0: tensor<8x1024x16x128xbf16>, %arg1: tensor<i32>) -> tensor<1024x16x128xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, %c, sizes = [1, 1024, 16, 128] : (tensor<8x1024x16x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1024x16x128xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x1024x16x128xbf16>) -> tensor<1024x16x128xbf16>
    return %1 : tensor<1024x16x128xbf16>
  }
  func.func private @dynamic_index_in_dim_6(%arg0: tensor<8x1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [1, 1] : (tensor<8x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x1xi32>
    %1 = stablehlo.reshape %0 : (tensor<1x1xi32>) -> tensor<1xi32>
    return %1 : tensor<1xi32>
  }
  func.func private @dynamic_index_in_dim_7(%arg0: tensor<8x1x10xi32>, %arg1: tensor<i32>) -> tensor<1x10xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, sizes = [1, 1, 10] : (tensor<8x1x10xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x10xi32>
    %1 = stablehlo.reshape %0 : (tensor<1x1x10xi32>) -> tensor<1x10xi32>
    return %1 : tensor<1x10xi32>
  }
  func.func private @dynamic_index_in_dim_8(%arg0: tensor<8x10x8x1x128xbf16>, %arg1: tensor<i32>) -> tensor<10x8x1x128xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, %c, %c, sizes = [1, 10, 8, 1, 128] : (tensor<8x10x8x1x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x10x8x1x128xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x10x8x1x128xbf16>) -> tensor<10x8x1x128xbf16>
    return %1 : tensor<10x8x1x128xbf16>
  }
  func.func private @closed_call(%arg0: tensor<1x16xi32>, %arg1: tensor<1x16xi32>, %arg2: tensor<1x16x1024xbf16>, %arg3: tensor<1024x6144xbf16>, %arg4: tensor<1024x6144xbf16>, %arg5: tensor<6144x1024xbf16>, %arg6: tensor<1024xbf16>, %arg7: tensor<1024xbf16>, %arg8: tensor<1024x8x128xbf16>, %arg9: tensor<128xbf16>, %arg10: tensor<16x128x1024xbf16>, %arg11: tensor<1024x16x128xbf16>, %arg12: tensor<128xbf16>, %arg13: tensor<1024x8x128xbf16>, %arg14: tensor<1xi32>, %arg15: tensor<1x10xi32>, %arg16: tensor<10x8x1x128xbf16>, %arg17: tensor<1xi32>, %arg18: tensor<10x8x1x128xbf16>) -> (tensor<1x16x1024xbf16>, tensor<1xi32>, tensor<1x10xi32>, tensor<1x16xi32>, tensor<10x8x1x128xbf16>, tensor<1xi32>, tensor<10x8x1x128xbf16>, tensor<16x8x1x128xbf16>, tensor<16x8x1x128xbf16>) {
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %cst_3 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_5 = stablehlo.constant dense<1.000000e+06> : tensor<f32>
    %c_6 = stablehlo.constant dense<2> : tensor<i32>
    %cst_7 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_9 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %cst_10 = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg2 : (tensor<1x16x1024xbf16>) -> tensor<1x16x1024xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<1x16x1024xf32>
    %2 = stablehlo.reduce(%1 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x16x1024xf32>, tensor<f32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %4 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<1x16x1xf32>
    %6 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x16x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<1x16x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1024xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<1x16x1024xf32>
    %11 = stablehlo.convert %10 : (tensor<1x16x1024xf32>) -> tensor<1x16x1024xbf16>
    %12 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<1024xbf16>
    %13 = stablehlo.add %arg7, %12 : tensor<1024xbf16>
    %14 = stablehlo.dot_general %13, %11, batching_dims = [0] x [2], contracting_dims = [] x [] : (tensor<1024xbf16>, tensor<1x16x1024xbf16>) -> tensor<1024x1x16xbf16>
    %15 = stablehlo.transpose %14, dims = [1, 2, 0] : (tensor<1024x1x16xbf16>) -> tensor<1x16x1024xbf16>
    %16 = sdy.sharding_constraint %15 <@mesh, [{}, {}, {}]> : tensor<1x16x1024xbf16>
    %17 = sdy.sharding_constraint %15 <@mesh, [{}, {}, {}]> : tensor<1x16x1024xbf16>
    %18 = stablehlo.dot_general %16, %arg11, contracting_dims = [2] x [0] : (tensor<1x16x1024xbf16>, tensor<1024x16x128xbf16>) -> tensor<1x16x16x128xbf16>
    %19 = stablehlo.dot_general %17, %arg8, contracting_dims = [2] x [0] : (tensor<1x16x1024xbf16>, tensor<1024x8x128xbf16>) -> tensor<1x16x8x128xbf16>
    %20 = stablehlo.dot_general %17, %arg13, contracting_dims = [2] x [0] : (tensor<1x16x1024xbf16>, tensor<1024x8x128xbf16>) -> tensor<1x16x8x128xbf16>
    %21 = stablehlo.convert %18 : (tensor<1x16x16x128xbf16>) -> tensor<1x16x16x128xf32>
    %22 = stablehlo.multiply %21, %21 : tensor<1x16x16x128xf32>
    %23 = stablehlo.reduce(%22 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x16x16x128xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<1x16x16xf32>) -> tensor<1x16x16x1xf32>
    %25 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x16x16x1xf32>
    %26 = stablehlo.divide %24, %25 : tensor<1x16x16x1xf32>
    %27 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x16x16x1xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x16x16x1xf32>
    %29 = stablehlo.rsqrt %28 : tensor<1x16x16x1xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x128xf32>
    %31 = stablehlo.multiply %21, %30 : tensor<1x16x16x128xf32>
    %32 = stablehlo.convert %31 : (tensor<1x16x16x128xf32>) -> tensor<1x16x16x128xbf16>
    %33 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<128xbf16>
    %34 = stablehlo.add %arg12, %33 : tensor<128xbf16>
    %35 = stablehlo.dot_general %34, %32, batching_dims = [0] x [3], contracting_dims = [] x [] : (tensor<128xbf16>, tensor<1x16x16x128xbf16>) -> tensor<128x1x16x16xbf16>
    %36 = stablehlo.transpose %35, dims = [1, 2, 3, 0] : (tensor<128x1x16x16xbf16>) -> tensor<1x16x16x128xbf16>
    %37 = stablehlo.convert %19 : (tensor<1x16x8x128xbf16>) -> tensor<1x16x8x128xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<1x16x8x128xf32>
    %39 = stablehlo.reduce(%38 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x16x8x128xf32>, tensor<f32>) -> tensor<1x16x8xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<1x16x8xf32>) -> tensor<1x16x8x1xf32>
    %41 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x16x8x1xf32>
    %42 = stablehlo.divide %40, %41 : tensor<1x16x8x1xf32>
    %43 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x16x8x1xf32>
    %44 = stablehlo.add %42, %43 : tensor<1x16x8x1xf32>
    %45 = stablehlo.rsqrt %44 : tensor<1x16x8x1xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x16x8x1xf32>) -> tensor<1x16x8x128xf32>
    %47 = stablehlo.multiply %37, %46 : tensor<1x16x8x128xf32>
    %48 = stablehlo.convert %47 : (tensor<1x16x8x128xf32>) -> tensor<1x16x8x128xbf16>
    %49 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<128xbf16>
    %50 = stablehlo.add %arg9, %49 : tensor<128xbf16>
    %51 = stablehlo.dot_general %50, %48, batching_dims = [0] x [3], contracting_dims = [] x [] : (tensor<128xbf16>, tensor<1x16x8x128xbf16>) -> tensor<128x1x16x8xbf16>
    %52 = stablehlo.transpose %51, dims = [1, 2, 3, 0] : (tensor<128x1x16x8xbf16>) -> tensor<1x16x8x128xbf16>
    %53 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1x1xi32>
    %54 = stablehlo.iota dim = 0 : tensor<64xi32>
    %55 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %56 = stablehlo.multiply %55, %54 : tensor<64xi32>
    %57 = stablehlo.convert %56 : (tensor<64xi32>) -> tensor<64xf32>
    %58 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %59 = stablehlo.divide %57, %58 : tensor<64xf32>
    %60 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %61 = stablehlo.power %60, %59 : tensor<64xf32>
    %62 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %63 = stablehlo.multiply %62, %61 : tensor<64xf32>
    %64 = stablehlo.convert %53 : (tensor<1x16x1x1xi32>) -> tensor<1x16x1x1xf32>
    %65 = stablehlo.broadcast_in_dim %63, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %66 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x64xf32>
    %67 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x16x1x64xf32>
    %68 = stablehlo.divide %66, %67 : tensor<1x16x1x64xf32>
    %69 = stablehlo.sine %68 : tensor<1x16x1x64xf32>
    %70 = stablehlo.convert %69 : (tensor<1x16x1x64xf32>) -> tensor<1x16x1x64xbf16>
    %71 = stablehlo.cosine %68 : tensor<1x16x1x64xf32>
    %72 = stablehlo.convert %71 : (tensor<1x16x1x64xf32>) -> tensor<1x16x1x64xbf16>
    %73 = stablehlo.concatenate %70, %70, dim = 3 : (tensor<1x16x1x64xbf16>, tensor<1x16x1x64xbf16>) -> tensor<1x16x1x128xbf16>
    %74 = stablehlo.concatenate %72, %72, dim = 3 : (tensor<1x16x1x64xbf16>, tensor<1x16x1x64xbf16>) -> tensor<1x16x1x128xbf16>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x16x1x128xbf16>) -> tensor<1x16x16x128xbf16>
    %76 = stablehlo.multiply %36, %75 : tensor<1x16x16x128xbf16>
    %77 = stablehlo.slice %36 [0:1, 0:16, 0:16, 0:64] : (tensor<1x16x16x128xbf16>) -> tensor<1x16x16x64xbf16>
    %78 = stablehlo.slice %36 [0:1, 0:16, 0:16, 64:128] : (tensor<1x16x16x128xbf16>) -> tensor<1x16x16x64xbf16>
    %79 = stablehlo.negate %78 : tensor<1x16x16x64xbf16>
    %80 = stablehlo.concatenate %79, %77, dim = 3 : (tensor<1x16x16x64xbf16>, tensor<1x16x16x64xbf16>) -> tensor<1x16x16x128xbf16>
    %81 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x16x1x128xbf16>) -> tensor<1x16x16x128xbf16>
    %82 = stablehlo.multiply %80, %81 : tensor<1x16x16x128xbf16>
    %83 = stablehlo.add %76, %82 : tensor<1x16x16x128xbf16>
    %84 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1x1xi32>
    %85 = stablehlo.iota dim = 0 : tensor<64xi32>
    %86 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %87 = stablehlo.multiply %86, %85 : tensor<64xi32>
    %88 = stablehlo.convert %87 : (tensor<64xi32>) -> tensor<64xf32>
    %89 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %90 = stablehlo.divide %88, %89 : tensor<64xf32>
    %91 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %92 = stablehlo.power %91, %90 : tensor<64xf32>
    %93 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %94 = stablehlo.multiply %93, %92 : tensor<64xf32>
    %95 = stablehlo.convert %84 : (tensor<1x16x1x1xi32>) -> tensor<1x16x1x1xf32>
    %96 = stablehlo.broadcast_in_dim %94, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %97 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x64xf32>
    %98 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x16x1x64xf32>
    %99 = stablehlo.divide %97, %98 : tensor<1x16x1x64xf32>
    %100 = stablehlo.sine %99 : tensor<1x16x1x64xf32>
    %101 = stablehlo.convert %100 : (tensor<1x16x1x64xf32>) -> tensor<1x16x1x64xbf16>
    %102 = stablehlo.cosine %99 : tensor<1x16x1x64xf32>
    %103 = stablehlo.convert %102 : (tensor<1x16x1x64xf32>) -> tensor<1x16x1x64xbf16>
    %104 = stablehlo.concatenate %101, %101, dim = 3 : (tensor<1x16x1x64xbf16>, tensor<1x16x1x64xbf16>) -> tensor<1x16x1x128xbf16>
    %105 = stablehlo.concatenate %103, %103, dim = 3 : (tensor<1x16x1x64xbf16>, tensor<1x16x1x64xbf16>) -> tensor<1x16x1x128xbf16>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2, 3] : (tensor<1x16x1x128xbf16>) -> tensor<1x16x8x128xbf16>
    %107 = stablehlo.multiply %52, %106 : tensor<1x16x8x128xbf16>
    %108 = stablehlo.slice %52 [0:1, 0:16, 0:8, 0:64] : (tensor<1x16x8x128xbf16>) -> tensor<1x16x8x64xbf16>
    %109 = stablehlo.slice %52 [0:1, 0:16, 0:8, 64:128] : (tensor<1x16x8x128xbf16>) -> tensor<1x16x8x64xbf16>
    %110 = stablehlo.negate %109 : tensor<1x16x8x64xbf16>
    %111 = stablehlo.concatenate %110, %108, dim = 3 : (tensor<1x16x8x64xbf16>, tensor<1x16x8x64xbf16>) -> tensor<1x16x8x128xbf16>
    %112 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x16x1x128xbf16>) -> tensor<1x16x8x128xbf16>
    %113 = stablehlo.multiply %111, %112 : tensor<1x16x8x128xbf16>
    %114 = stablehlo.add %107, %113 : tensor<1x16x8x128xbf16>
    %115 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<bf16>) -> tensor<1x16x16x128xbf16>
    %116 = stablehlo.multiply %83, %115 : tensor<1x16x16x128xbf16>
    %117 = sdy.sharding_constraint %116 <@mesh, [{}, {}, {}, {}]> : tensor<1x16x16x128xbf16>
    %118 = sdy.sharding_constraint %114 <@mesh, [{}, {}, {}, {}]> : tensor<1x16x8x128xbf16>
    %119 = sdy.sharding_constraint %20 <@mesh, [{}, {}, {}, {}]> : tensor<1x16x8x128xbf16>
    %120 = stablehlo.transpose %118, dims = [1, 2, 0, 3] : (tensor<1x16x8x128xbf16>) -> tensor<16x8x1x128xbf16>
    %121 = stablehlo.transpose %119, dims = [1, 2, 0, 3] : (tensor<1x16x8x128xbf16>) -> tensor<16x8x1x128xbf16>
    %122 = stablehlo.reshape %117 : (tensor<1x16x16x128xbf16>) -> tensor<1x16x8x2x128xbf16>
    %123 = stablehlo.dot_general %118, %122, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4] : (tensor<1x16x8x128xbf16>, tensor<1x16x8x2x128xbf16>) -> tensor<1x8x16x16x2xbf16>
    %124 = stablehlo.transpose %123, dims = [0, 1, 4, 3, 2] : (tensor<1x8x16x16x2xbf16>) -> tensor<1x8x2x16x16xbf16>
    %125 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1xi32>
    %126 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<1x16xi32>) -> tensor<1x1x16xi32>
    %127 = stablehlo.broadcast_in_dim %125, dims = [0, 1, 2] : (tensor<1x16x1xi32>) -> tensor<1x16x16xi32>
    %128 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2] : (tensor<1x1x16xi32>) -> tensor<1x16x16xi32>
    %129 = stablehlo.compare EQ, %127, %128, SIGNED : (tensor<1x16x16xi32>, tensor<1x16x16xi32>) -> tensor<1x16x16xi1>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 3, 4] : (tensor<1x16x16xi1>) -> tensor<1x1x1x16x16xi1>
    %131 = stablehlo.iota dim = 0 : tensor<16x16xi32>
    %132 = stablehlo.iota dim = 1 : tensor<16x16xi32>
    %133 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %134 = stablehlo.add %131, %133 : tensor<16x16xi32>
    %135 = stablehlo.compare LE, %132, %134, SIGNED : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %136 = stablehlo.broadcast_in_dim %135, dims = [3, 4] : (tensor<16x16xi1>) -> tensor<1x1x1x16x16xi1>
    %137 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<1x1x1x16x16xi1>
    %138 = stablehlo.compare NE, %130, %137, UNSIGNED : (tensor<1x1x1x16x16xi1>, tensor<1x1x1x16x16xi1>) -> tensor<1x1x1x16x16xi1>
    %139 = stablehlo.convert %138 : tensor<1x1x1x16x16xi1>
    %140 = stablehlo.and %139, %136 : tensor<1x1x1x16x16xi1>
    %141 = call @_where(%140, %cst_11, %cst_1) : (tensor<1x1x1x16x16xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x16x16xf32>
    %142 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1x16x16xf32>
    %143 = stablehlo.compare GE, %141, %142, FLOAT : (tensor<1x1x1x16x16xf32>, tensor<1x1x1x16x16xf32>) -> tensor<1x1x1x16x16xi1>
    %144 = call @_where_9(%143, %124, %cst_1) : (tensor<1x1x1x16x16xi1>, tensor<1x8x2x16x16xbf16>, tensor<f32>) -> tensor<1x8x2x16x16xbf16>
    %145 = stablehlo.reshape %144 : (tensor<1x8x2x16x16xbf16>) -> tensor<1x16x16x16xbf16>
    %146 = stablehlo.reduce(%145 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x16x16xbf16>, tensor<bf16>) -> tensor<1x16x16xbf16>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2] : (tensor<1x16x16xbf16>) -> tensor<1x16x16x1xbf16>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2, 3] : (tensor<1x16x16x1xbf16>) -> tensor<1x16x16x16xbf16>
    %149 = stablehlo.subtract %145, %148 : tensor<1x16x16x16xbf16>
    %150 = stablehlo.exponential %149 : tensor<1x16x16x16xbf16>
    %151 = stablehlo.convert %150 : (tensor<1x16x16x16xbf16>) -> tensor<1x16x16x16xf32>
    %152 = stablehlo.reduce(%151 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x16x16x16xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x16x16xf32>) -> tensor<1x16x16x1xf32>
    %154 = stablehlo.convert %153 : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xbf16>
    %155 = stablehlo.reshape %150 : (tensor<1x16x16x16xbf16>) -> tensor<1x8x2x16x16xbf16>
    %156 = stablehlo.transpose %154, dims = [0, 2, 1, 3] : (tensor<1x16x16x1xbf16>) -> tensor<1x16x16x1xbf16>
    %157 = stablehlo.dot_general %119, %155, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4] : (tensor<1x16x8x128xbf16>, tensor<1x8x2x16x16xbf16>) -> tensor<1x8x128x2x16xbf16>
    %158 = stablehlo.transpose %157, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x16xbf16>) -> tensor<1x16x8x2x128xbf16>
    %159 = stablehlo.reshape %158 : (tensor<1x16x8x2x128xbf16>) -> tensor<1x16x16x128xbf16>
    %160 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x16x16x1xbf16>) -> tensor<1x16x16x128xbf16>
    %161 = stablehlo.divide %159, %160 : tensor<1x16x16x128xbf16>
    %162 = sdy.sharding_constraint %161 <@mesh, [{}, {}, {}, {}]> : tensor<1x16x16x128xbf16>
    %163 = stablehlo.dot_general %162, %arg10, contracting_dims = [2, 3] x [0, 1] : (tensor<1x16x16x128xbf16>, tensor<16x128x1024xbf16>) -> tensor<1x16x1024xbf16>
    %164 = stablehlo.add %arg2, %163 : tensor<1x16x1024xbf16>
    %165 = stablehlo.convert %164 : (tensor<1x16x1024xbf16>) -> tensor<1x16x1024xf32>
    %166 = stablehlo.multiply %165, %165 : tensor<1x16x1024xf32>
    %167 = stablehlo.reduce(%166 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x16x1024xf32>, tensor<f32>) -> tensor<1x16xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %169 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %170 = stablehlo.divide %168, %169 : tensor<1x16x1xf32>
    %171 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %172 = stablehlo.add %170, %171 : tensor<1x16x1xf32>
    %173 = stablehlo.rsqrt %172 : tensor<1x16x1xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1024xf32>
    %175 = stablehlo.multiply %165, %174 : tensor<1x16x1024xf32>
    %176 = stablehlo.convert %175 : (tensor<1x16x1024xf32>) -> tensor<1x16x1024xbf16>
    %177 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<1024xbf16>
    %178 = stablehlo.add %arg6, %177 : tensor<1024xbf16>
    %179 = stablehlo.dot_general %178, %176, batching_dims = [0] x [2], contracting_dims = [] x [] : (tensor<1024xbf16>, tensor<1x16x1024xbf16>) -> tensor<1024x1x16xbf16>
    %180 = stablehlo.transpose %179, dims = [1, 2, 0] : (tensor<1024x1x16xbf16>) -> tensor<1x16x1024xbf16>
    %181 = stablehlo.dot_general %180, %arg3, contracting_dims = [2] x [0] : (tensor<1x16x1024xbf16>, tensor<1024x6144xbf16>) -> tensor<1x16x6144xbf16>
    %182 = call @silu(%181) : (tensor<1x16x6144xbf16>) -> tensor<1x16x6144xbf16>
    %183 = stablehlo.dot_general %180, %arg4, contracting_dims = [2] x [0] : (tensor<1x16x1024xbf16>, tensor<1024x6144xbf16>) -> tensor<1x16x6144xbf16>
    %184 = stablehlo.multiply %182, %183 : tensor<1x16x6144xbf16>
    %185 = sdy.sharding_constraint %184 <@mesh, [{}, {}, {}]> : tensor<1x16x6144xbf16>
    %186 = stablehlo.dot_general %185, %arg5, contracting_dims = [2] x [0] : (tensor<1x16x6144xbf16>, tensor<6144x1024xbf16>) -> tensor<1x16x1024xbf16>
    %187 = stablehlo.add %164, %186 : tensor<1x16x1024xbf16>
    return %187, %arg14, %arg15, %arg1, %arg16, %arg17, %arg18, %120, %121 : tensor<1x16x1024xbf16>, tensor<1xi32>, tensor<1x10xi32>, tensor<1x16xi32>, tensor<10x8x1x128xbf16>, tensor<1xi32>, tensor<10x8x1x128xbf16>, tensor<16x8x1x128xbf16>, tensor<16x8x1x128xbf16>
  }
  func.func private @_where(%arg0: tensor<1x1x1x16x16xi1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1x1x1x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1x16x16xf32>
    %2 = stablehlo.select %arg0, %0, %1 : tensor<1x1x1x16x16xi1>, tensor<1x1x1x16x16xf32>
    return %2 : tensor<1x1x1x16x16xf32>
  }
  func.func private @_where_9(%arg0: tensor<1x1x1x16x16xi1>, %arg1: tensor<1x8x2x16x16xbf16>, %arg2: tensor<f32>) -> tensor<1x8x2x16x16xbf16> {
    %0 = stablehlo.convert %arg2 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x16x16xi1>) -> tensor<1x8x2x16x16xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<1x8x2x16x16xbf16>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<1x8x2x16x16xi1>, tensor<1x8x2x16x16xbf16>
    return %3 : tensor<1x8x2x16x16xbf16>
  }
  func.func private @silu(%arg0: tensor<1x16x6144xbf16>) -> tensor<1x16x6144xbf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %0 = stablehlo.negate %arg0 : tensor<1x16x6144xbf16>
    %1 = stablehlo.exponential %0 : tensor<1x16x6144xbf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x16x6144xbf16>
    %3 = stablehlo.add %2, %1 : tensor<1x16x6144xbf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x16x6144xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<1x16x6144xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x16x6144xbf16>
    return %6 : tensor<1x16x6144xbf16>
  }
  func.func private @dynamic_update_index_in_dim(%arg0: tensor<8x1xi32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> tensor<8x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<8x1xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<8x1xi32>
    return %1 : tensor<8x1xi32>
  }
  func.func private @dynamic_update_index_in_dim_10(%arg0: tensor<8x1x10xi32>, %arg1: tensor<1x10xi32>, %arg2: tensor<i32>) -> tensor<8x1x10xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<1x10xi32>) -> tensor<1x1x10xi32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c : (tensor<8x1x10xi32>, tensor<1x1x10xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x10xi32>
    return %1 : tensor<8x1x10xi32>
  }
  func.func private @dynamic_update_index_in_dim_11(%arg0: tensor<8x1x16xi32>, %arg1: tensor<1x16xi32>, %arg2: tensor<i32>) -> tensor<8x1x16xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<1x16xi32>) -> tensor<1x1x16xi32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c : (tensor<8x1x16xi32>, tensor<1x1x16xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x16xi32>
    return %1 : tensor<8x1x16xi32>
  }
  func.func private @dynamic_update_index_in_dim_12(%arg0: tensor<8x10x8x1x128xbf16>, %arg1: tensor<10x8x1x128xbf16>, %arg2: tensor<i32>) -> tensor<8x10x8x1x128xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2, 3, 4] : (tensor<10x8x1x128xbf16>) -> tensor<1x10x8x1x128xbf16>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c, %c, %c : (tensor<8x10x8x1x128xbf16>, tensor<1x10x8x1x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x10x8x1x128xbf16>
    return %1 : tensor<8x10x8x1x128xbf16>
  }
  func.func private @dynamic_update_index_in_dim_13(%arg0: tensor<8x16x8x1x128xbf16>, %arg1: tensor<16x8x1x128xbf16>, %arg2: tensor<i32>) -> tensor<8x16x8x1x128xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2, 3, 4] : (tensor<16x8x1x128xbf16>) -> tensor<1x16x8x1x128xbf16>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c, %c, %c : (tensor<8x16x8x1x128xbf16>, tensor<1x16x8x1x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x16x8x1x128xbf16>
    return %1 : tensor<8x16x8x1x128xbf16>
  }
  func.func private @argmax(%arg0: tensor<1x1x2048xf32>) -> tensor<1x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.iota dim = 2 : tensor<1x1x2048xi32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [2] : (tensor<1x1x2048xf32>, tensor<1x1x2048xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x1xf32>, tensor<1x1xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare GT, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare EQ, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare LT, %arg2, %arg4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
    }
    return %1#1 : tensor<1x1xi32>
  }
}
