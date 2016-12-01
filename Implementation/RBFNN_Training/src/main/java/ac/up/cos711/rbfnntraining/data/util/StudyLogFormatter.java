package ac.up.cos711.rbfnntraining.data.util;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Date;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;

/**
 * Custom log formatter  
 * 
 * @author Abrie van Aardt
 */
public class StudyLogFormatter extends Formatter {

    private final String formatString = "%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS Thread %6$02d %2$-7s %3$-40s %4$s %5$s%n";

    @Override
    public synchronized String format(LogRecord record) {
        Date date = new Date();
        date.setTime(record.getMillis());

        String sourceClassName = record.getSourceClassName();

        if (sourceClassName == null)
            sourceClassName = record.getLoggerName();

        sourceClassName = sourceClassName.substring(sourceClassName.lastIndexOf('.') + 1);
        String sourceMethodName = record.getSourceMethodName();
        String logLevel = record.getLevel().getLocalizedName();
        String message = formatMessage(record);
        String thrown = "";

        if (record.getThrown() != null) {
            StringWriter sw = new StringWriter();
            try (PrintWriter pw = new PrintWriter(sw)) {
                record.getThrown().printStackTrace(pw);
                thrown = System.lineSeparator() + sw.toString();
            }
        }

        String formattedLogEntry = String.format(formatString,
                date,
                logLevel,
                sourceClassName + "." + sourceMethodName,
                message,
                thrown,
                record.getThreadID());

        return formattedLogEntry;
    }
}
