# softrender — 纯 CPU 软件光栅化渲染器

一个不依赖 OpenGL / Vulkan / DirectX / GPU 或任何外部渲染服务的软件渲染器。
所有模型、纹理、帧缓冲和输出图像只存在于本地内存或文件中，全部计算在 CPU
上完成（仅依赖 `numpy` 做矩阵运算，图像输出仅用 Python 标准库）。

## 目录结构

```
softrender/
  math3d.py      # 4x4 矩阵：model/view/projection/viewport 变换
  rasterizer.py  # 顶点变换、近平面裁剪、光栅化、z-buffer、透视校正插值
  texture.py     # 纹理存储与 nearest / bilinear 采样（clamp / repeat）
  imageio.py     # PPM (P6) 与 PNG 输出（PNG 用标准库 zlib 手工编码）
tests/           # 自动化测试（pytest，全部终端执行，不开窗口）
examples/render_demo.py  # 渲染演示场景到 out/demo.ppm / out/demo.png
```

## 渲染流水线

每个三角形依次经过：

1. **顶点变换**：`clip = P · V · M · [x, y, z, 1]ᵀ`，即模型（model）→
   视图（view）→ 投影（projection）矩阵级联，得到齐次裁剪坐标。
2. **近平面裁剪**：在齐次裁剪空间对近平面做 Sutherland–Hodgman 裁剪，
   保留半空间 `z ≥ -w`（即视图空间 `z ≤ -near`）。完全位于近平面后方的
   三角形被丢弃；部分穿过的三角形被切成 1–2 个新三角形，新顶点的位置与
   属性（uv、颜色）在裁剪空间线性插值——因为发生在透视除法之前，该
   插值是精确的。裁剪后 `w > 0` 恒成立，不会产生无效坐标。
3. **透视除法 + 视口变换**：`ndc = clip.xyz / clip.w`，再映射到屏幕像素
   坐标（y 轴向下）：`sx = (ndc.x + 1) · W/2`，`sy = (1 − ndc.y) · H/2`。
4. **光栅化**：对包围盒内每个像素中心 `(px+0.5, py+0.5)` 计算重心坐标
   （barycentric coordinates），三个权重均 ≥ 0 则像素被覆盖。
5. **深度测试**：NDC 的 z 在屏幕空间是线性的，直接按重心坐标插值；
   与 z-buffer 比较，更小（更近）的深度获胜，与三角形提交顺序无关。
6. **透视校正插值 + 纹理采样**：按 1/w 加权插值顶点属性，采样纹理并与
   顶点颜色相乘，写入帧缓冲。

## 坐标空间约定

- 右手坐标系，列向量，`p' = M · p`。
- 相机（视图空间）看向 **−Z** 方向（OpenGL 约定）。
- 透视投影矩阵把视锥映射到 NDC `[-1, 1]³`（近平面 → z = −1，远平面 → z = +1）。
- 屏幕空间原点位于左上角，y 轴向下，像素中心采样。

## 插值公式

设像素重心坐标为 `(λ₀, λ₁, λ₂)`，三个顶点的裁剪空间 w 为 `wᵢ`，属性为 `Aᵢ`。

- **深度（线性量）**：`z = Σ λᵢ · zᵢ`（NDC z 在屏幕空间本身就是线性的）。
- **透视校正属性插值**：

  ```
  A = ( Σ λᵢ · Aᵢ / wᵢ ) / ( Σ λᵢ / wᵢ )
  ```

  即先以 `1/wᵢ` 为权重做加权平均再归一化。若直接在屏幕空间线性插值
  `Σ λᵢ · Aᵢ`，纹理坐标会在倾斜表面上产生明显拉伸错误——测试
  `tests/test_perspective.py` 用视空间射线-平面求交得到的真实值验证了
  该公式的正确性（误差 < 1e-6），并证明朴素线性插值偏差 > 0.05。

## 纹理采样

- **nearest**：取 `floor(u·W), floor(v·H)` 处纹素。
- **bilinear**：纹素 `(i, j)` 中心位于 `((i+0.5)/W, (j+0.5)/H)`，对相邻
  四个纹素做双线性插值。
- **边界处理**：`clamp` 模式钳制到边缘纹素（uv 超出 [0,1] 不越界）；
  `repeat` 模式取模平铺。

## 运行测试

```bash
pip install numpy pytest   # 仅这两个依赖
python3 -m pytest tests/ -v
```

测试全部在终端执行，不打开任何窗口，覆盖：

- `test_transform.py` — 模型/视图/投影/视口矩阵链的数值正确性
- `test_raster.py` — 重心坐标与三角形像素覆盖、顶点颜色插值
- `test_depth.py` — 重叠三角形的 z-buffer 遮挡（与绘制顺序无关）
- `test_perspective.py` — 透视校正插值对标射线求交真值
- `test_clip.py` — 近平面裁剪（完全后方丢弃 / 部分穿过切分 / 属性插值）
- `test_texture.py` — nearest / bilinear 采样与 clamp / repeat 边界
- `test_output.py` — 渲染固定场景，输出 PPM/PNG 并校验固定像素值与
  SHA-256 哈希

## 渲染演示

```bash
python3 examples/render_demo.py   # 生成 out/demo.ppm 和 out/demo.png
```

演示场景包含透视棋盘格地板（透视校正插值 + bilinear 采样 + repeat 平铺）
和两个互相重叠的三角形（近处绿色遮挡远处红色，验证 z-buffer）。
