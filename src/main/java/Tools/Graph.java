package Tools;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.List;

public class Graph {
    public static void buildGraphRW_Vs(List<Double> V, int currentEpoch){
        for (int i = 0; i < V.size(); i++) { System.out.println((i+1)+": " + V.get(i)); }
        XYSeries series = new XYSeries("V(s)");
        for (int i = 0; i < V.size(); i++) { series.add((i+1), V.get(i)); }
        XYDataset xyDataset = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory
                .createXYLineChart("y = V(s)", "state number", "avg score",
                        xyDataset,
                        PlotOrientation.VERTICAL,
                        true, true, true);
        JFrame frame = new JFrame("Value function for random walk, " + (currentEpoch) + " epoch");
        frame.getContentPane()
                .add(new ChartPanel(chart));
        frame.setSize(1000,800);
        frame.show();
    }
}
