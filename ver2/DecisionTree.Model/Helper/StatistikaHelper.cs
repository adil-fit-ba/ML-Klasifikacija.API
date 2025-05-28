using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Helper;

public static class StatistikaHelper
{
    public static double IzracunajStandardnuDevijaciju(List<double> vrijednosti, double srednja)
    {
        return Math.Sqrt(vrijednosti.Average(x => Math.Pow(x - srednja, 2)));
    }

    public static (double min, double max, double srednja, double stdDev) Statistika(List<double> vrijednosti)
    {
        var srednja = vrijednosti.Average();
        return (vrijednosti.Min(), vrijednosti.Max(), srednja, IzracunajStandardnuDevijaciju(vrijednosti, srednja));
    }
}

