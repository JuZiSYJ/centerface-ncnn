# centerface-ncnn
using ncnn, on android,detection,surport **arm v7/v8/x86**, surport **vulkan** on gpu

## environment
[ncnn](https://github.com/Tencent/ncnn)  
AndroidStudio3.2(NDK19)  
Qualcomm845


## model
using pretrained **centerface** ,by[star-clouds](https://github.com/Star-Clouds/CenterFace)   
**pipeline**:  model on **PC -->   ONNX  ---> model on ncnn(.bin,,param) --> inference on Android**  

## cost_time(only once)
item | PC(i7 8700 500*300) |  Android 模拟 600*400 
-|-|-
load_model | 5ms | 35ms |
infer_time | 192ms | 190ms |
decode_time | 0ms | 1ms |

## organization
using **ncnn** and **opencv** on android  
the sim-overall android-organization is here

![organization](./test-img/buju.png)

## result
infer time on **845** in float32 is aroud ***350ms(700x500)***, depending on the object  
![845-cpu](./test-img/Screenshot_2020-06-02-21-13-29-537_com.example.sy.jpg)
![845-gpu](./test-img/Screenshot_2020-06-02-21-13-57-663_com.example.sy.jpg)

**ncnn** is good in optimazing on float32

## reference
(https://github.com/JuZiSYJ/MobilenetSSD_Android)

(https://github.com/nihui/ncnn-android-squeezenet)

(https://blog.csdn.net/qq_33431368/article/details/84990390)
