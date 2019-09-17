# glumpy_obj_render

We need to render 3D model in python at running-time. 
For example in instance-level 3D pose tracking and result demonstration in 3D pose detection.

[SIXD toolkit](https://github.com/thodan/sixd_toolkit) has implemented render scripts using glumpy.
In the toolkit the model used is ply and both depth ang rgb image are rendered.

However, I only want to get the mask of the object or the depth image. So I only render depth image.
Only vertex and faces information are need from 3D model files to render depth img.
So I write a simple script to load vertex and faces from .obj files,
The depth render part is modified from  [SIXD toolkit](https://github.com/thodan/sixd_toolkit).

And the test code used [RBot dataset](http://cvmr.mi.hs-rm.de/research/RBOT/).