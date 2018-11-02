package Tools;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
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

    public static void buildPolicy2D(final String[][] data, final int w, final String Title){
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame.setDefaultLookAndFeelDecorated(true);
                createGUI(Title, w, data);
            }
        });
    }
    public static void createGUI(String title, int w, String[][] data) {
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        String[] columnNames = new String[w];
        for (int i = 0; i < w; i++) { columnNames[i] = ""; }

        JTable table = new JTable(data, columnNames);

        JScrollPane scrollPane = new JScrollPane(table);

        frame.getContentPane().add(scrollPane);
        frame.setPreferredSize(new Dimension(300, 400));
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
