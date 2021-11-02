import SimpleITK as sitk
import sys

if len(sys.argv) < 2:
    print("Usage: DicomImagePrintTags <input_file>")
    sys.exit(1)

reader = sitk.ImageFileReader()

reader.SetFileName(sys.argv[1])
reader.LoadPrivateTagsOn()

reader.ReadImageInformation()

for k in reader.GetMetaDataKeys():
    v = reader.GetMetaData(k)
    print(f"({k}) = = \"{v}\"")

print(f"Image Size: {reader.GetSize()}")
print(f"Image PixelType: {sitk.GetPixelIDValueAsString(reader.GetPixelID())}")

# The Code above this point views the metadata for a specified fil
# The user specifies the file in the terminal