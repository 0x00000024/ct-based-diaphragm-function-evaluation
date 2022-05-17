import {SciChart3DSurface} from "scichart/Charting3D/Visuals/SciChart3DSurface";
import {NumericAxis3D} from "scichart/Charting3D/Visuals/Axis/NumericAxis3D";
import {CameraController} from "scichart/Charting3D/CameraController";
import {Vector3} from "scichart/Charting3D/Vector3";
import {OrbitModifier3D} from "scichart/Charting3D/ChartModifiers/OrbitModifier3D";
import {MouseWheelZoomModifier3D} from "scichart/Charting3D/ChartModifiers/MouseWheelZoomModifier3D";
import {XyzDataSeries3D} from "scichart/Charting3D/Model/DataSeries/XyzDataSeries3D";
import {ScatterRenderableSeries3D} from "scichart/Charting3D/Visuals/RenderableSeries/ScatterRenderableSeries3D";
import {PixelPointMarker3D} from "scichart/Charting3D/Visuals/PointMarkers/DefaultPointMarkers";
import {SciChartSurface} from "scichart";
import {EAutoRange} from "scichart/types/AutoRange";

const jsonData = require('./json/data.json');

class Data {
  constructor(jsonData) {
    this.jsonData = jsonData;
    this.x = this.getDataFromObj2Array('x');
    this.y = this.getDataFromObj2Array('y');
    this.z = this.getDataFromObj2Array('z');
  }

  getDataFromObj2Array(axis) {
    const axis_data = [];
    for (const [_, value] of Object.entries(this.jsonData[axis])) {
      axis_data.push(value);
    }
    return axis_data;
  }
}

async function initSciChart() {
  // Create the SciChart3DSurface in the div 'scichart-root'
  // The SciChart3DSurface, and webassembly context 'wasmContext' are paired. This wasmContext
  // instance must be passed to other types that exist on the same surface.
  SciChartSurface.setRuntimeLicenseKey();

  const {wasmContext, sciChart3DSurface} = await SciChart3DSurface.create(
    'scichart-root'
  );

  // Create and attach a camera to the 3D Viewport
  sciChart3DSurface.camera = new CameraController(wasmContext, {
    position: new Vector3(300, 300, 300),
    target: new Vector3(0, 50, 0),
  });

  // Add an X,Y,Z axis to the viewport
  sciChart3DSurface.xAxis = new NumericAxis3D(wasmContext, {
    axisTitle: 'X Axis',
  });
  sciChart3DSurface.xAxis.autoRange = EAutoRange.Always
  sciChart3DSurface.yAxis = new NumericAxis3D(wasmContext, {
    axisTitle: 'Y Axis',
  });
  sciChart3DSurface.yAxis.autoRange = EAutoRange.Always
  sciChart3DSurface.zAxis = new NumericAxis3D(wasmContext, {
    axisTitle: 'Z Axis',
  });
  sciChart3DSurface.zAxis.autoRange = EAutoRange.Always

  // Create a 3D Scatter series using pixel point marker, a high performance single pixel applied per x,y,z data-point
  const data = new Data(jsonData)
  const xyzDataSeries = new XyzDataSeries3D(wasmContext);
  if (data !== undefined) {
    xyzDataSeries.appendRange(data.x, data.y, data.z);
  }

  const series = new ScatterRenderableSeries3D(wasmContext, {
    pointMarker: new PixelPointMarker3D(wasmContext, {fill: '#00FF00'}),
    dataSeries: xyzDataSeries,
  });
  sciChart3DSurface.renderableSeries.add(series);

  sciChart3DSurface.xAxis = new NumericAxis3D(wasmContext);
  sciChart3DSurface.yAxis = new NumericAxis3D(wasmContext);
  sciChart3DSurface.zAxis = new NumericAxis3D(wasmContext);

  // That's it! You just created your first SciChart3DSurface!
  sciChart3DSurface.chartModifiers.add(new MouseWheelZoomModifier3D());
  sciChart3DSurface.chartModifiers.add(new OrbitModifier3D());
}

initSciChart();
