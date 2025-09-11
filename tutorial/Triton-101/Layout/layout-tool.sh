triton-tensor-layout -l "#ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>" -t "tensor<128x256xf16>"
                     -o output_mma.txt

triton-tensor-layout -l "#ttg.blocked<{sizePerThread=[2,2], threadsPerWarp=[8,4], warpsPerCTA=[1,2], order=[1,0]}>" -t "tensor<16x16xf16>" \
                     -o output_blocked_layout_tensor_view.txt

triton-tensor-layout -l "#ttg.blocked<{sizePerThread=[2,2], threadsPerWarp=[8,4], warpsPerCTA=[1,2], order=[1,0]}>" -t "tensor<16x16xf16>" \
                     -o output_blocked_layout_hw_view.txt \
                     -use-hw-view

triton-tensor-layout -l "#ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>" -t "tensor<4x4xf16>" \
                     -o output_swizzled_shared_layout.txt

triton-tensor-layout -l "#ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>" -t "tensor<4x4xf16>" \
                     -o output_swizzled_shared_layout2.txt

triton-tensor-layout -l "#ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>" -t "tensor<8x4xf16>" \
                     -o output_swizzled_shared_layout2.txt  

# 目前不支持
# triton-tensor-layout -l "#ttg.blocked_layout<{, \                   
#   ,sizePerThread = {2, 2} \
#   ,threadsPerWarp = {8, 4} \
#   ,warpsPerCTA = {1, 2} \
#   ,CTAsPerCGA = {1, 1} \
#   ,CTASplitNum = {1, 1} \
# }>" -t "tensor<32x32xf16>" \
#                      -o output_blocked_layout_verbose.txt

triton-tensor-layout -l "#ttg.swizzled_shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>" -t "tensor<4x8xf16>" \
                     -o output_swizzled_shared_layout3.txt

triton-tensor-layout -i input.mlir -t "tensor<1x128x128xf16>" -o output_tensor_view.txt

triton-tensor-layout -i input.mlir -t "tensor<1x128x128xf16>" -o output_hw_view.txt -alias-names="blocked,mma" -use-hw-view