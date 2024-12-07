from plyfile import PlyData

ply_path = "/media/zhaoyibin/common/3DRE/3DGS/MVSGaussian/3dgs_next/scan105/scan105.ply"
plydata = PlyData.read(ply_path)
print(plydata)
