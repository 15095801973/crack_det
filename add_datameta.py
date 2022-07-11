from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os

""" ... """
"""Creates the metadata for an image classifier."""

# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "concrete crack detection"
model_meta.description = ("混凝土表面裂缝无损检测和分割")
model_meta.version = "v1"
model_meta.author = "LR"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")

# Creates input info.
input_meta_image = _metadata_fb.TensorMetadataT()
input_meta_image_meta = _metadata_fb.TensorMetadataT()
input_meta_anchor = _metadata_fb.TensorMetadataT()


# Creates output info.
output_meta_detection = _metadata_fb.TensorMetadataT()
output_meta_class = _metadata_fb.TensorMetadataT()
output_meta_bbox = _metadata_fb.TensorMetadataT()
output_meta_mask= _metadata_fb.TensorMetadataT()
output_meta_ROI = _metadata_fb.TensorMetadataT()
output_meta_rpn_class = _metadata_fb.TensorMetadataT()
output_meta_rpn_box = _metadata_fb.TensorMetadataT()


input_meta_image.name = "image"
input_meta_image.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(128, 128))
input_meta_image.content = _metadata_fb.ContentT()
input_meta_image.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta_image.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta_image.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5]
input_normalization.options.std = [127.5]
input_meta_image.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta_image.stats = input_stats


input_meta_image_meta.name = "image_meta"
input_meta_image_meta.description = (
    "meta_disc")
input_meta_image_meta.content = _metadata_fb.ContentT()
input_meta_image_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
input_meta_image_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

input_meta_anchor.name = "anchor"
input_meta_anchor.description = (
    "anchor_disc")
input_meta_anchor.content = _metadata_fb.ContentT()
input_meta_anchor.content.contentProperties = _metadata_fb.FeaturePropertiesT()
input_meta_anchor.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)


output_meta_detection.name = "output_meta_detection"
output_meta_detection.description=("output_meta_detection_disc")
output_meta_detection.content = _metadata_fb.ContentT()
output_meta_detection.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_detection.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

output_meta_class.name = "output_meta_class"
output_meta_class.content = _metadata_fb.ContentT()
output_meta_class.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_class.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

output_meta_bbox.name = "output_meta_bbox"
output_meta_bbox.content = _metadata_fb.ContentT()
output_meta_bbox.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_bbox.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

output_meta_mask.name = "output_meta_mask"
output_meta_mask.description=("output_meta_mask")
output_meta_mask.content = _metadata_fb.ContentT()
output_meta_mask.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_mask.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

output_meta_ROI.name = "output_meta_ROI"
output_meta_ROI.content = _metadata_fb.ContentT()
output_meta_ROI.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_ROI.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

output_meta_rpn_class.name = "output_meta_rpn_class"
output_meta_rpn_class.content = _metadata_fb.ContentT()
output_meta_rpn_class.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_rpn_class.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

output_meta_rpn_box.name = "output_meta_rpn_box"
output_meta_rpn_box.content = _metadata_fb.ContentT()
output_meta_rpn_box.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta_rpn_box.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)


# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 1001 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = "your_path_to_label_file" #os.path.basename("your_path_to_label_file")
label_file.description = "Labels for objects that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]


# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta_image, input_meta_image_meta, input_meta_anchor]
subgraph.outputTensorMetadata = [output_meta_detection,output_meta_mask]
# output_meta_class,
# output_meta_bbox,

# output_meta_mask,

# output_meta_ROI,
# output_meta_rpn_class,
# output_meta_rpn_box]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

populator = _metadata.MetadataPopulator.with_model_file("mask_rcnn_crack_test.tflite")
populator.load_metadata_buffer(metadata_buf)
# populator.load_associated_files(["your_path_to_label_file"])
populator.populate()

export_model_path = "mask_rcnn_crack_test.tflite"
export_json_file = "metadata.json"
displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
# export_json_file = os.path.join(FLAGS.export_directory,
#                     os.path.splitext(model_basename)[0] + ".json")
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file
with open(export_json_file, "w") as f:
  f.write(json_file)