package ac.up.cos711.rbfnntraining.data;

import ac.up.cos711.rbfnntraining.data.util.GraphException;
import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.function.problem.RealProblem;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Wraps the GNUPlot process and serves as an adapter to route graphing
 * requests. This class only exposes GNUPlot functionality that will be needed
 * during the course of the project.
 *
 * @author Abrie van Aardt
 */
public class Graph {

    /**
     * Inner class used to represent a single plot on the graph.
     */
    public class Plot {

        public String title;
        public String dataset;
        public String type;
        public String colour;
    }

    /**
     * Initialises a graph with the properties given as parameters. Other
     * properties are either fixed or derived from the parameters given.
     *
     * @param _path the relative directory path the graph should be placed
     * @param _title the title of the graph
     * @param xLabel x-axis label
     * @param yLabel y-axis label
     * @param zLabel z-axis label
     * @param _dimensions the number of dimensions the graph will be plotted in
     * @throws GraphException
     */
    public Graph(String _path, String _title, String xLabel, String yLabel, String zLabel, int _dimensions) throws GraphException {
        try {
            gnuPlotProcess = Runtime.getRuntime().exec("gnuplot --persist");
            gnuPlotTerminal = new BufferedWriter(new OutputStreamWriter(gnuPlotProcess.getOutputStream()));
            //ensure the directory where the graph will reside, exists
            Files.createDirectories(Paths.get(_path, _title));
        }
        catch (IOException e) {
            throw new GraphException(e.getMessage());
        }

        path = _path;
        title = _title;
        properties = new ArrayList<>(10);
        plots = new ArrayList<>();
        plotCommand = "";

        dimensions = _dimensions;

        //set some graph properties
        //Terminal
        //properties.add("set terminal postscript eps enhanced color font 'Helvetica,16'");
        //Output
        //properties.add("set output '" + ".\\" + path + "\\" + title + "\\" + title + ".eps'");
        //Title
        properties.add("set title '" + title + "'");

        //XLabel
        properties.add("set xlabel '" + xLabel + "'");

        //YLabel
        properties.add("set ylabel '" + yLabel + "'");

        //Auto scale axes
        properties.add("set auto fix");

        //Leave space around the graph
        properties.add("set offsets graph 0.03, graph 0.03, graph 0.03, graph 0.03");

        //Use Colour Palette
//        properties.add("load 'config\\gnuplot-color-palettes\\paired.pal'");

        //set colours to classic
        properties.add("set colors classic");

        //Settings for bar charts
        properties.add("set boxwidth 0.5");        
        properties.add("set style fill solid");

        switch (dimensions) {
            case 2:
                break;
            case 3:
                //ZLabel
                properties.add("set zlabel '" + zLabel + "'");
                properties.add("unset colorbox");
                properties.add("set palette grey");
                break;
            default:
                throw new GraphException("Can only plot in 2 or 3 dimensions");
        }

    }

    /**
     * Adds a plot of the problem to the graph, using the problem's defined
     * bounds to confine the plot.
     *
     * @param _title the title of the plot which will be used as key
     * @param problem the function to plot
     * @throws GraphException
     */
    public void addPlot(String _title, RealProblem problem)
            throws GraphException {
        addPlot(_title, problem, problem.getLowerBound(), problem.getUpperBound());
    }

    /**
     * Adds a plot of the function to the graph using the upper and lower 
     * bounds that are provided as parameters.
     *
     * @param _title the title of the plot which will be used as key
     * @param function the function to plot
     * @param lowerbound the lower bound for all but the last axis
     * @param upperbound the upper bound for all but the last axis
     * @throws GraphException
     */
    public void addPlot(String _title, Function function, double lowerbound, double upperbound)
            throws GraphException {

        if (lowerbound == Double.NEGATIVE_INFINITY
                || upperbound == Double.POSITIVE_INFINITY)
            throw new GraphException("Cannot plot between infinite bounds.");

        Plot plot = new Plot();
       
         //if no plot title, assume only single plot and name it the same 
        //as the graph
        if (_title == null || _title == ""){
            _title = title;
            plot.title = "";
        } else{
            plot.title = _title;
        }  
        
        plot.type = dimensions == 2 ? "lines" : "pm3d";       
        
        plot.dataset = "'.\\" + path + "\\" + title + "\\" + _title + ".dat'";

        final int detail = 250;

        try {
            BufferedWriter writer = new BufferedWriter(
                    new FileWriter(path + "\\" + title + "\\" + _title + ".dat", false));

            if (dimensions == 2) {
                for (int i = 0; i <= detail; i++) {
                    double x = lowerbound + i * (upperbound - lowerbound) / detail;
                    double y = function.evaluate(x);
                    writer.write(Double.toString(x));
                    writer.write(" " + Double.toString(y));
                    writer.newLine();
                }
            }
            else if (dimensions == 3) {
                for (int i = 0; i <= detail; i++) {
                    double x = lowerbound + i * (upperbound - lowerbound) / detail;
                    for (int j = 0; j <= detail; j++) {
                        double y = lowerbound + j * (upperbound - lowerbound) / detail;
                        writer.write(Double.toString(x));
                        writer.write(" " + Double.toString(y));
                        writer.write(" " + Double.toString(function.evaluate(x, y)));
                        writer.newLine();
                    }
                    writer.newLine();//next datablock
                }
            }

            writer.flush();
            writer.close();
            plots.add(plot);
        }
        catch (UnequalArgsDimensionException | IOException e) {
            throw new GraphException(e.getMessage());
        }

    }

    /**
     * Adds a 2D plot of the data points given as parameters.
     *
     * @param _title the title of the plot
     * @param xData array containing x coordinates
     * @param yData array containing y coordinates
     * @throws GraphException
     */
    public void addPlot(String _title, double[] xData, double[] yData, String type)
            throws GraphException {

           
        Plot plot = new Plot();
       
         //if no plot title, assume only single plot and name it the same 
        //as the graph
        if (_title == null || _title == ""){
            _title = title;
            plot.title = "";
        } else{
            plot.title = _title;
        }    
        
        plot.type = type;        

        plot.dataset = "'.\\" + path + "\\" + title + "\\" + _title + ".dat'";

        if (xData.length != yData.length)
            throw new GraphException("X and Y arrays must be of same length to plot.");

        try {
            BufferedWriter writer = new BufferedWriter(
                    new FileWriter(path + "\\" + title + "\\" + _title + ".dat", false));

            for (int i = 0; i < xData.length; i++) {
                writer.write(Double.toString(xData[i]));
                writer.write(" " + Double.toString(yData[i]));
                writer.newLine();
            }

            writer.flush();
            writer.close();
            plots.add(plot);
        }
        catch (IOException e) {
            throw new GraphException(e.getMessage());
        }
    }

    /**
     * Adds a 1/2D plot of the data points given as parameters.
     *
     * @param _title the title of the plot
     * @param xData 2D array containing x coordinates (1D/2D)
     * @param yData array containing y coordinates
     * @throws GraphException
     */
    public void addPlot(String _title, double[][] xData, double[] yData, String type)
            throws GraphException {

        if (xData[0].length != dimensions - 1)
            throw new GraphException("Plot has to be the same dimensionality as the graph.");

        Plot plot = new Plot();
        plot.title = _title;
        plot.type = type;
        plot.dataset = "'.\\" + path + "\\" + title + "\\" + _title + ".dat'";

        if (xData.length != yData.length)
            throw new GraphException("X and Y arrays must be of same length to plot.");

        try {
            BufferedWriter writer = new BufferedWriter(
                    new FileWriter(path + "\\" + title + "\\" + _title + ".dat", false));

            for (int i = 0; i < xData.length; i++) {
                for (int j = 0; j < xData[i].length; j++) {
                    writer.write(Double.toString(xData[i][j]) + " ");
                }
                writer.write(Double.toString(yData[i]));
                writer.newLine();
            }

            writer.flush();
            writer.close();
            plots.add(plot);
        }
        catch (IOException e) {
            throw new GraphException(e.getMessage());
        }
    }

    /**
     * Tells GNUPlot to draw all the plots that have been added to the graph
     * with the properties specified.
     *
     * @throws GraphException
     */
    public void plot() throws GraphException {

        if (dimensions == 2) {
            plotCommand = "plot";
        }
        else {
            plotCommand = "splot";
        }

        //use with palettes: " ls " + (i + 1) + 
        for (int i = 0; i < plots.size(); i++) {
            plotCommand += " " + plots.get(i).dataset + " title '" + plots.get(i).title + "' with " + plots.get(i).type + ",";
        }

        try {
            for (String property : properties) {
                gnuPlotTerminal.write(property);
                gnuPlotTerminal.newLine();
            }

            gnuPlotTerminal.write(plotCommand);
            gnuPlotTerminal.newLine();
            gnuPlotTerminal.flush();            
            gnuPlotTerminal.close();            
            
            gnuPlotProcess.waitFor();
        }
        catch (IOException | InterruptedException e) {
            throw new GraphException("There was an error communicating with GNUPlot: " + e.getMessage());
        }

        plotCommand = "";
    }

    /**
     * Adds the property to the graph. This can also be used to add a property
     * that effectively unsets a previous property.
     *
     * @param property
     */
    public void addProperty(String property) {
        properties.add(property);
    }

    private List<String> properties;
    private int dimensions;
    private String path;
    private String title;
    private List<Plot> plots;
    private String plotCommand;
    private final transient Process gnuPlotProcess;
    private final transient BufferedWriter gnuPlotTerminal;

}
