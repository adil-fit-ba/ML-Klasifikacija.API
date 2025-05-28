using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Helper;

public static class DistribucijaHelper
{
    public static List<(string vrijednost, int count)> TopVrijednosti(List<string> vrijednosti, int topN = 5)
    {
        return vrijednosti
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .GroupBy(v => v)
            .OrderByDescending(g => g.Count())
            .Take(topN)
            .Select(g => (g.Key, g.Count()))
            .ToList();
    }
}
