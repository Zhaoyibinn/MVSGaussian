import numpy as np
import argparse
import re

def read_ply_header(f):
    """读取PLY文件头部，返回头部内容和顶点数量"""
    header = []
    while True:
        line = f.readline()
        # 尝试用utf-8解码，如果失败则用latin-1（更宽容的编码）
        try:
            decoded_line = line.decode('utf-8').rstrip('\r\n')
        except UnicodeDecodeError:
            decoded_line = line.decode('latin-1').rstrip('\r\n')
        header.append(decoded_line)
        if decoded_line == 'end_header':
            break
    
    # 提取顶点数量
    vertex_count = None
    for line in header:
        match = re.match(r'element vertex (\d+)', line)
        if match:
            vertex_count = int(match.group(1))
            break
    
    if vertex_count is None:
        raise ValueError("无法从PLY头部找到顶点数量")
    
    return '\n'.join(header) + '\n', vertex_count

def merge_ply_files(file1, file2, file3, output_file):
    """合并三个具有相同属性的二进制PLY点云文件"""
    # 读取所有文件的数据
    all_data = []
    headers = []
    
    for filename in [file1, file2, file3]:
        with open(filename, 'rb') as f:
            # 读取头部
            header, vertex_count = read_ply_header(f)
            headers.append(header)
            
            # 验证所有文件的头部结构相同（除了顶点数量）
            if len(all_data) > 0:
                # 比较头部，忽略顶点数量行
                header_lines = [line for line in header.split('\n') if not line.startswith('element vertex')]
                ref_header_lines = [line for line in headers[0].split('\n') if not line.startswith('element vertex')]
                if header_lines != ref_header_lines:
                    raise ValueError(f"文件 {filename} 的头部结构与第一个文件不同，无法合并")
            
            # 计算剩余需要读取的字节数
            # 每个顶点有17个float32属性
            bytes_to_read = vertex_count * 17 * 4  # 17个属性 × 4字节/float32
            data = np.fromfile(f, dtype=np.float32, count=vertex_count * 17)
            all_data.append(data)
    
    # 合并所有数据
    merged_data = np.concatenate(all_data, axis=0)
    total_vertices = merged_data.shape[0] // 17
    print(f"合并完成，总顶点数: {total_vertices}")
    
    # 写入合并后的文件
    with open(output_file, 'wb') as f:
        # 写入头部（使用第一个文件的头部，更新顶点数量）
        header = headers[0]
        header = re.sub(r'element vertex \d+', f'element vertex {total_vertices}', header)
        f.write(header.encode('utf-8'))
        
        # 写入二进制数据
        merged_data.astype(np.float32).tofile(f)
    
    print(f"合并后的文件已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并三个具有相同属性的二进制PLY点云文件')
    parser.add_argument('--file1', default="test_0.ply")
    parser.add_argument('--file2', default="test_1.ply")
    parser.add_argument('--file3', default="test_2.ply")
    parser.add_argument('--output', default='test_merged.ply')
    args = parser.parse_args()
    
    merge_ply_files(args.file1, args.file2, args.file3, args.output)
    