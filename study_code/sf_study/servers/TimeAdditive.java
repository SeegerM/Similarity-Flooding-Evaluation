// X51 (EXPLORATORY): the cost of the additive term's arithmetic, S = max(q, 0) + 5 * D, on the cluster host itself
// (the hosts have no numpy; this is plain Java on the hosts' JDK, i.e. how Schematch would compute it in-process).
// Standalone: it reads the FILE seed matrices (Schematch's CSV layout: header row, row label in column 0) and touches
// nothing of Schematch. Per scenario: parse time of both files, and the median of 7 timed repetitions of the
// arithmetic after 3 untimed warm-ups.
//
//   javac TimeAdditive.java && java -cp . TimeAdditive <seeds root> <q matcher> <D matcher> > additive_times.tsv
//   output: dataset \t scenario \t rows \t cols \t parse_ns \t arith_ns_median
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class TimeAdditive {
    static double[][] read(Path f) throws IOException {
        List<String> lines = Files.readAllLines(f);
        double[][] m = new double[lines.size() - 1][];
        for (int i = 1; i < lines.size(); i++) {
            String[] p = lines.get(i).split(",", -1);
            m[i - 1] = new double[p.length - 1];
            for (int j = 1; j < p.length; j++) m[i - 1][j - 1] = Double.parseDouble(p[j]);
        }
        return m;
    }

    public static void main(String[] a) throws IOException {
        Path root = Paths.get(a[0]);
        Path qRoot = root.resolve(a[1]), dRoot = root.resolve(a[2]);
        List<Path> seeds;
        try (Stream<Path> s = Files.walk(qRoot)) {
            seeds = s.filter(p -> p.getFileName().toString().equals("seed.csv")).sorted().collect(Collectors.toList());
        }
        double sink = 0;
        for (Path fq : seeds) {
            Path rel = qRoot.relativize(fq);
            Path fd = dRoot.resolve(rel);
            if (!Files.isRegularFile(fd)) continue;
            long t0 = System.nanoTime();
            double[][] q = read(fq), d = read(fd);
            long parse = System.nanoTime() - t0;
            int r = q.length, c = q[0].length;
            long[] t = new long[7];
            double[][] s = new double[r][c];
            for (int k = -3; k < 7; k++) {
                long t1 = System.nanoTime();
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < c; j++) s[i][j] = Math.max(q[i][j], 0.0) + 5.0 * d[i][j];
                long dt = System.nanoTime() - t1;
                if (k >= 0) t[k] = dt;
                sink += s[r - 1][c - 1];
            }
            Arrays.sort(t);
            System.out.println(rel.getName(0) + "\t" + rel.getName(1) + "\t" + r + "\t" + c + "\t" + parse + "\t" + t[3]);
        }
        System.err.println("checksum " + sink);
    }
}
