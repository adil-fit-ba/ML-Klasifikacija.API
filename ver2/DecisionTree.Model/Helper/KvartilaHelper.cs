using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Helper
{
    public class KvartilaHelper
    {
        public static (double q1, double q3) IzracunajKvartile(List<RedPodatka> podaci, string nazivKolone)
        {
            List<double> vrijednosti = podaci
                .Select(p => p.Atributi.TryGetValue(nazivKolone, out var v) ? v?.Broj : null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .OrderBy(x => x)
                .ToList();

            if (!vrijednosti.Any())
                throw new InvalidOperationException($"Kolona '{nazivKolone}' ne sadrži numeričke vrijednosti.");

            int count = vrijednosti.Count;
            double q1 = vrijednosti[count / 4];
            double q3 = vrijednosti[3 * count / 4];

            return (q1, q3);
        }

        public static List<double> KvantilniThresholdovi(List<double> vrijednosti, int brojGrupa)
        {
            var sortirano = vrijednosti.OrderBy(x => x).ToList();
            var thresholds = new List<double>();

            for (int i = 1; i < brojGrupa; i++)
            {
                double kvantil = i / (double)brojGrupa;
                int indeks = (int)(kvantil * sortirano.Count);
                thresholds.Add(sortirano[indeks]);
            }

            return thresholds.Distinct().ToList();
        }
    }
}
