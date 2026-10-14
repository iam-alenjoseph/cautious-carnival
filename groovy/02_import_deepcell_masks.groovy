/**
 * 02_import_deepcell_masks.groovy
 * 
 * Deep Learning Cell Segmentation Pipeline
 * 
 * Purpose:
 *   Imports DeepCell (Mesmer) segmentation masks (.tif) back into QuPath,
 *   converting labeled regions into native PathCellObject instances with both
 *   nuclear and whole-cell boundaries positioned in global WSI coordinates.
 *
 * Requirements:
 *   - QuPath 0.4.x or 0.5.x
 *   - ImageJ-backed mask reading or QuPath Bio-Formats
 *
 * Usage:
 *   1. Open the original slide image in QuPath.
 *   2. Set `masksDir` below to the directory containing DeepCell output masks
 *      and the corresponding `tiles_metadata.json`.
 *   3. Run this script in QuPath Script Editor.
 */

import qupath.lib.images.ImageData
import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI
import qupath.imagej.tools.IJTools
import ij.IJ
import ij.ImagePlus
import ij.process.ImageProcessor
import ij.plugin.filter.ThresholdToSelection
import groovy.json.JsonSlurper
import java.io.File

// ==========================================
// CONFIGURATION
// ==========================================
def masksDir = new File("c:/Users/micke/OneDrive/Documents/DEEPCELL/data/segmentation_output")
def clearExistingDetections = false   // Set true to remove prior cell detections

// ==========================================
// INITIALIZATION
// ==========================================
def imageData = getCurrentImageData()
if (imageData == null) {
    println "[ERROR] No image currently open in QuPath."
    return
}

def hierarchy = imageData.getHierarchy()
if (clearExistingDetections) {
    def cells = hierarchy.getCellObjects()
    hierarchy.removeObjects(cells, true)
    println "[INFO] Removed ${cells.size()} existing cell objects."
}

File metaFile = new File(masksDir, "tiles_metadata.json")
if (!metaFile.exists()) {
    println "[WARN] Metadata file not found at ${metaFile.getAbsolutePath()}. Proceeding with direct mask scan..."
}

def metadata = metaFile.exists() ? new JsonSlurper().parse(metaFile) : null
def maskFiles = masksDir.listFiles().findAll { 
    it.name.endsWith("_whole_cell_mask.tif") || it.name.endsWith("_cell_mask.tif") 
}

println "=========================================================="
println "Importing DeepCell Masks from: ${masksDir.getAbsolutePath()}"
println "Total Mask Files Identified: ${maskFiles.size()}"
println "=========================================================="

List<PathCellObject> newCells = []
int globalCellCounter = 0

maskFiles.eachWithIndex { cellMaskFile, idx ->
    String baseName = cellMaskFile.name.replace("_whole_cell_mask.tif", "").replace("_cell_mask.tif", "")
    File nucMaskFile = new File(masksDir, "${baseName}_nuclear_mask.tif")

    // Retrieve tile origin offset (X, Y) from metadata or filename pattern
    int offsetX = 0
    int offsetY = 0
    double downsample = 1.0

    if (metadata != null && metadata.tiles != null) {
        def tileInfo = metadata.tiles.find { it.tile_id == baseName || cellMaskFile.name.startsWith(it.tile_id) }
        if (tileInfo != null) {
            offsetX = (int) tileInfo.x
            offsetY = (int) tileInfo.y
            downsample = (double) (tileInfo.downsample ?: 1.0)
        }
    } else {
        // Fallback: extract x and y from filename pattern like '*_x1024_y2048*'
        def matcher = (baseName =~ /_x(\d+)_y(\d+)/)
        if (matcher.find()) {
            offsetX = matcher.group(1).toInteger()
            offsetY = matcher.group(2).toInteger()
        }
    }

    ImagePlus impCell = IJ.openImage(cellMaskFile.getAbsolutePath())
    ImagePlus impNuc = nucMaskFile.exists() ? IJ.openImage(nucMaskFile.getAbsolutePath()) : null

    if (impCell == null) {
        println "[WARN] Could not open cell mask: ${cellMaskFile.name}"
        return
    }

    ImageProcessor ipCell = impCell.getProcessor()
    ImageProcessor ipNuc = impNuc?.getProcessor()

    // Find all unique cell label IDs (excluding background = 0)
    def cellLabels = new HashSet<Integer>()
    int width = ipCell.getWidth()
    int height = ipCell.getHeight()

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = (int) ipCell.getf(x, y)
            if (val > 0) cellLabels.add(val)
        }
    }

    println "Processing tile [${idx + 1}/${maskFiles.size()}]: ${baseName} with ${cellLabels.size()} cells (Offset: X=${offsetX}, Y=${offsetY})"

    // Convert each labeled region into a QuPath ROI
    cellLabels.each { labelVal ->
        // Generate binary threshold for whole cell
        ImageProcessor binCell = ipCell.duplicate()
        binCell.setThreshold(labelVal, labelVal, ImageProcessor.NO_LUT_UPDATE)
        ij.gui.Roi ijCellRoi = ThresholdToSelection.run(new ImagePlus("bin", binCell))

        if (ijCellRoi != null) {
            ROI qupathCellRoi = IJTools.convertToROI(ijCellRoi, cal)
            
            // Translate ROI to global slide coordinate space
            if (downsample != 1.0 || offsetX != 0 || offsetY != 0) {
                qupathCellRoi = RoiTools.translate(qupathCellRoi, offsetX, offsetY)
            }

            ROI qupathNucRoi = null
            if (ipNuc != null) {
                ImageProcessor binNuc = ipNuc.duplicate()
                binNuc.setThreshold(labelVal, labelVal, ImageProcessor.NO_LUT_UPDATE)
                ij.gui.Roi ijNucRoi = ThresholdToSelection.run(new ImagePlus("nuc", binNuc))
                if (ijNucRoi != null) {
                    qupathNucRoi = IJTools.convertToROI(ijNucRoi, cal)
                    if (downsample != 1.0 || offsetX != 0 || offsetY != 0) {
                        qupathNucRoi = RoiTools.translate(qupathNucRoi, offsetX, offsetY)
                    }
                }
            }

            // Create PathCellObject
            PathCellObject cellObj = PathObjects.createCellObject(qupathCellRoi, qupathNucRoi, null, null)
            cellObj.setName("Cell_${++globalCellCounter}")
            newCells.add(cellObj)
        }
    }

    impCell.close()
    impNuc?.close()
}

// Add imported cells to QuPath hierarchy
hierarchy.addPathObjects(newCells)
println "[SUCCESS] Successfully imported ${newCells.size()} cell objects into QuPath hierarchy!"

