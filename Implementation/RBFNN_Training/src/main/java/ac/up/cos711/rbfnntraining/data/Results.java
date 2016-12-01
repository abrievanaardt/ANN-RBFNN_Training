package ac.up.cos711.rbfnntraining.data;

import ac.up.cos711.rbfnntraining.data.util.GraphException;
import ac.up.cos711.rbfnntraining.data.util.ResultsException;
import ac.up.cos711.rbfnntraining.experiment.GDExperiment;
import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.function.problem.RealProblem;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Allows for experiment data to be written to file and/or directly transformed
 * into plots using GNUPlot. To support multi-threaded experiments and still
 * maintain separation of concerns, this class handles communication with 
 * {@linkGraph} behalf of the {@link GDExperiment} class.
 *
 * @author Abrie van Aardt
 */
public class Results {

    /**
     * Appends the value(s) to a file with the relative path:
     * experimentPath/resultName
     *
     * @param experimentPath
     * @param resultName
     * @param values
     * @throws ResultsException
     */
    public static void writeToFile(String experimentPath, String resultName, double... values) throws ResultsException {
        BufferedWriter writer = null;
        try {
            File directory = new File(experimentPath + "/" + resultName + ".csv");
            directory.getParentFile().mkdirs();
            writer = new BufferedWriter(new FileWriter(directory));

            for (int i = 0; i < values.length; i++) {
                writer.write(Double.toString(values[i]));
                writer.newLine();
            }
//TODO ADD PARAMETER
            writer.flush();           
            
        }
        catch (IOException e) {
            throw new ResultsException(e.getMessage());
        }
        finally {
            if (writer != null) {
                try {
                    writer.close();
                }
                catch (IOException e) {
                }
            }
        }
    }

    /**
     * Appends the value(s) to a file with the relative path:
     * experimentPath/resultName
     *
     * @param experimentPath
     * @param resultName
     * @param values1
     * @param values2
     * @throws ResultsException
     */
    public static void writeToFile(String experimentPath, String resultName, double[]... values) throws ResultsException {
        BufferedWriter writer = null;

        for (int i = 1; i < values.length; i++) {
            if (values[i - 1].length != values[i].length)
                throw new ResultsException("The length of the arrays do not correspond");
        }
        
        try {
            File directory = new File(experimentPath + "/" + resultName + ".csv");
            directory.getParentFile().mkdirs();
            writer = new BufferedWriter(new FileWriter(directory));

            for (int i = 0; i < values.length; i++) {

                String line = Double.toString(values[i][0]);
                for (int j = 1; j < values[i].length; j++) {
                    line += ", " + values[i][j];
                }
                writer.write(line);
                writer.newLine();
            }

            writer.flush();
        }
        catch (IOException e) {
            throw new ResultsException(e.getMessage());
        }
        finally {
            if (writer != null) {
                try {
                    writer.close();
                }
                catch (IOException e) {
                }
            }
        }
    }

    public static void newGraph(GDExperiment experiment, String path, String title, String xLabel, String yLabel, String zLabel, int dimensions)
            throws ResultsException {
        try {
            experiment.graph = new Graph(path, title, xLabel, yLabel, zLabel, dimensions);
        }
        catch (GraphException e) {
            throw new ResultsException(e.getMessage());
        }
    }

    public static void addPlot(GDExperiment experiment, String title, RealProblem problem)
            throws ResultsException {
        try {
            experiment.graph.addPlot(title, problem);
        }
        catch (GraphException e) {
            throw new ResultsException(e.getMessage());
        }
    }

    public static void addPlot(GDExperiment experiment, String title, Function function, int lowerbound, int upperbound)
            throws ResultsException {
        try {
            experiment.graph.addPlot(title, function, lowerbound, upperbound);
        }
        catch (GraphException e) {
            throw new ResultsException(e.getMessage());
        }
    }

    public static void addPlot(GDExperiment experiment, String title, double[] xData, double[] yData, String type)
            throws ResultsException {
        try {
            experiment.graph.addPlot(title, xData, yData, type);
        }
        catch (GraphException e) {
            throw new ResultsException(e.getMessage());
        }
    }

    public static void addPlot(GDExperiment experiment, String title, double[][] xData, double[] yData, String type)
            throws ResultsException {

        try {
            experiment.graph.addPlot(title, xData, yData, type);
        }
        catch (GraphException e) {
            throw new ResultsException(e.getMessage());
        }
    }

    public static void plot(GDExperiment experiment)
            throws ResultsException {
        try {
            experiment.graph.plot();
        }
        catch (GraphException e) {
            throw new ResultsException(e.getMessage());
        }
    }   
    
}
