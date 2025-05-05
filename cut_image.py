import nibabel as nib
import pandas as pd
import os

# Excel文件路径
excel_path = 'E:/data_set.xlsx'
df = pd.read_excel(excel_path)

# 遍历每行，自动处理每位患者
for index, row in df.iterrows():
    image_id = row['examID']  # 获取图像ID
    start_slice = row['Start_Slice']  # 获取起始层面
    end_slice = row['End_Slice']  # 获取结束层面

    # 图像路径
    img_path_img = f'G:/nii_gz/{image_id}.nii.gz'

    # 加载图像
    img = nib.load(img_path_img)
    img_data = img.get_fdata()

    # 检查层面范围是否在图像层数范围内
    if end_slice > img_data.shape[2]:
        print(f"图像 {image_id} 的层数不足 {end_slice}，跳过此图像")
        continue

    # 截取指定层面
    cropped_data = img_data[:, :, start_slice:end_slice]

    # 创建新的图像对象
    new_img = nib.Nifti1Image(cropped_data, img.affine, img.header)

    # 若覆盖原始图像，则注释本行
    output_dir = 'G:/nii_gz_cropped'
    cropped_img_path = os.path.join(output_dir, f'{image_id}_cropped.nii.img')

    nib.save(new_img, cropped_img_path) # 保存覆盖原始图像img_path_img，不覆盖cropped_img_path
    print(f"{index}. 图像 {image_id} 已截取并保存，层面范围：{start_slice}-{end_slice}")

