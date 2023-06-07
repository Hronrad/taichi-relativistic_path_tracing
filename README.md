# taichi-relativistic_path_tracing
## 项目目的
相对论效应下的真实场景是什么样的？过去有一些程序动画的模拟结果，都能很好地反映狭义相对论效应下的变形和谱移，但是其虚拟性总是是我不能很直观地感受如果发生在真实世界将会是什么样的效果。于是尝试自己写一个程序实现相对真实地渲染狭义相对论效应下的场景。
## 成果展示

<div>
  <img src="images/example_0.png" style="display: inline-block; width: 49%;">
  <img src="images/example_2.png" style="display: inline-block; width: 49%;">
</div>
<center>无相对运动</center>

![z轴相对运动](images/example_0.gif)
<center>相对运动（未添加多普勒效应）</center>

<div>
  <img src="images/example_0_1.gif" style="display: inline-block; width: 49%;">
  <img src="images/example_0_2.gif" style="display: inline-block; width: 49%;">
</div>

<center>z轴相对速度 beta = 0.4 时分别产生红移和蓝移 </center>

<div>
  <img src="images/example_0_3.gif" style="display: inline-block; width: 49%;">
  <img src="images/example_0_4.gif" style="display: inline-block; width: 49%;">
</div>
<center>横向和竖向相对运动时产生明显形变效应</center>

## 程序运行
需要用到的依赖库
```
pip install numpy
pip install taichi
```
taichi-pathtracer中搭建场景

运行程序
```
python3 ./taichi-pathtracer.py
```


## 原理介绍
### 狭义相对论视觉效应

### 多普勒频移

### 相对论光线追踪

## 