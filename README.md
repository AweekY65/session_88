# softrenderer — 纯 CPU 软件光栅化渲染器

一个从零实现的软件渲染管线。**不依赖 OpenGL / Vulkan / DirectX / GPU 或任何
外部渲染服务**：所有模型、纹理、帧缓冲都只存在于内存（numpy 数组）中，输出
图像写为本地 PPM / PNG 文件（PNG 写入器仅用 Python 标准库 `zlib`/`struct`
实现）。唯一的第三方依赖是 `numpy`（测试使用 `pytest`）。

## 目录结构

```
softrenderer/
  math3d.py      # 4x4 矩阵：translate/scale/rotate、look_at、perspective
  clip.py        # 齐次裁剪空间中的近平面裁剪（Sutherland-Hodgman）
  texture.py     # 纹理存储与采样（nearest / bilinear，clamp / repeat）
  rasterizer.py  # 帧缓冲、三角形光栅化、z-buffer、完整渲染管线
  imageio.py     # PPM(P6) 与 PNG 写出（纯标准库）
tests/           # pytest 自动化测试（全部终端执行，不开窗口）
examples/render_demo.py  # 渲染演示场景到 demo.ppm / demo.png
```

## 渲染流水线

每个三角形依次经过：

```
模型空间 --model--> 世界空间 --view--> 观察空间 --projection--> 裁剪空间
   --> 近平面裁剪 --> 透视除法(NDC) --> 视口变换(屏幕空间)
   --> 光栅化(重心坐标) --> 透视校正插值 --> z-buffer 深度测试 --> 纹理采样
```

对应代码：`Renderer.render()`（`softrenderer/rasterizer.py`）。

- **model / view / projection**：列向量约定，`clip = P @ V @ M @ [p, 1]`。
  `look_at` 构造视图矩阵，`perspective` 构造 OpenGL 风格投影矩阵。
- **近平面裁剪**：在齐次裁剪空间对平面 `z = -w` 做 Sutherland-Hodgman
  裁剪（在透视除法之前），因此位于相机后方（`w <= 0`）的顶点永远不会
  产生无效屏幕坐标。裁剪可能输出 0、1 或 2 个三角形，交点处的 uv /
  颜色属性在裁剪空间线性插值。
- **透视除法与视口变换**：`ndc = clip.xyz / clip.w`；
  `sx = (ndc.x+1)/2 * W`，`sy = (1-(ndc.y+1)/2) * H`（y 翻转使 +Y 向上），
  `sz = (ndc.z+1)/2`（0 = 近，1 = 远）。

## 坐标空间约定

- 右手坐标系；观察空间中相机位于原点，看向 **-Z**，+Y 向上，+X 向右。
- NDC 可见范围为 `x, y, z ∈ [-1, 1]`，`z = -1` 为近平面，`z = +1` 为远平面。
- 屏幕空间原点位于左上角，像素中心在半整数坐标 `(px+0.5, py+0.5)`。
- 纹理坐标 `(0,0)` 为纹理左上角，`u` 向右、`v` 向下。

## 光栅化与插值公式

**重心坐标**：对屏幕空间顶点 `v0, v1, v2` 与像素中心 `p`，用边函数
`E(a,b,p) = (p.x-a.x)(b.y-a.y) - (p.y-a.y)(b.x-a.x)` 计算

```
w0 = E(v1,v2,p)/E(v1,v2,v0),  w1 = E(v2,v0,p)/E(v1,v2,v0),  w2 = 1-w0-w1
```

三者均 `>= 0` 时像素被覆盖（与绕序无关）。

**深度**：`z/w` 在屏幕空间是线性的，因此直接重心插值
`z = Σ wi·zi`，与 z-buffer 比较（小者胜出），保证重叠三角形显示最近表面。

**透视校正插值**：属性 `a`（uv、顶点色）满足 `a/w` 在屏幕空间线性，故

```
a(p) = ( Σ wi·ai·(1/wi) ) / ( Σ wi·(1/wi) )
```

而不是简单的 `Σ wi·ai`（那会在倾斜表面上产生明显的纹理扭曲）。

**纹理采样**：texel 中心位于 `((i+0.5)/W, (j+0.5)/H)`。
`nearest` 取 `floor(u·W), floor(v·H)`；`bilinear` 对相邻四个 texel 做双线性
加权。越界坐标由 `clamp`（夹到边缘）或 `repeat`（取模重复）处理，纹理边界
行为定义明确。

## 运行测试

```bash
cd <仓库根目录>
python3 -m pytest tests -q
```

测试全部在终端执行、不打开任何窗口，覆盖：

- `tests/test_math3d.py` — 矩阵变换（model/view/projection 及完整 MVP 到屏幕坐标）
- `tests/test_rasterizer.py` — 三角形覆盖（像素与面积）、深度遮挡、
  透视校正插值（与解析光线/平面求交结果对比，并验证朴素线性插值确实偏差）、
  近平面裁剪（穿过近平面、完全在后面、顶点在相机平面上）
- `tests/test_clip.py` — 近平面裁剪的三角形数量、交点位置与属性插值
- `tests/test_texture.py` — nearest / bilinear 采样与 clamp / repeat 边界
- `tests/test_output.py` — 渲染小场景写出 PPM/PNG，校验固定像素、
  PNG  chunk/CRC/解压数据与帧缓冲一致，以及整帧 SHA-256 黄金哈希

## 渲染演示

```bash
python3 examples/render_demo.py [输出目录]
```

生成 `demo.ppm` 与 `demo.png`：棋盘格地面（透视校正插值）+ 三个相互重叠的
三角形（z-buffer 遮挡）。
