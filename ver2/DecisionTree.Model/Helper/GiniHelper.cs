using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Helper
{
    public class GiniHelper
    {
        public static string NajcescaKlasa(List<RedPodatka> podaci) =>
     podaci.GroupBy(p => p.Klasa)
           .OrderByDescending(g => g.Count())
           .First().Key;

        public static double IzracunajGiniIndeks(List<RedPodatka> podaci, string atribut)
        {
            var grupe = podaci.GroupBy(p => p.Atributi[atribut]);
            double ukupno = podaci.Count;
            double gini = 0.0;

            foreach (var grupa in grupe)
            {
                double velicina = grupa.Count();
                double skor = grupa.GroupBy(p => p.Klasa)
                                    .Select(g => Math.Pow(g.Count() / velicina, 2))
                                    .Sum();
                gini += (1 - skor) * (velicina / ukupno);
            }

            return gini;
        }

        public static double IzracunajGiniIndeks(List<RedPodatka> podaci)
        {
            if (podaci.Count == 0)
                return 0;

            double ukupno = podaci.Count;
            var grupe = podaci.GroupBy(p => p.Klasa);
            double skor = grupe.Select(g => Math.Pow(g.Count() / ukupno, 2)).Sum();
            return 1 - skor;
        }

        public static (double NajmanjiGini, double NajboljiThreshold) IzracunajGiniSaGrupama(List<RedPodatka> podaci, string atribut, int brojGrupa = 100)
        {
            var vrijednosti = podaci
                .Select(p => p.Atributi[atribut].Broj)
                .Where(b => b.HasValue)
                .Select(b => b.Value)
                .OrderBy(v => v)
                .ToList();

            if (vrijednosti.Count < 2)
                return (double.MaxValue, double.NaN);

            int grupaVelicina = vrijednosti.Count / brojGrupa;
            var pragovi = new List<double>();

            for (int i = 1; i < brojGrupa; i++)
            {
                int index = i * grupaVelicina;
                if (index < vrijednosti.Count)
                {
                    double threshold = vrijednosti[index];
                    pragovi.Add(threshold);
                }
            }

            double najboljiGini = double.MaxValue;
            double najboljiThreshold = double.NaN;

            foreach (var threshold in pragovi.Distinct())
            {
                var lijevi = podaci
                    .Where(p => p.Atributi[atribut].Broj.HasValue && p.Atributi[atribut].Broj.Value <= threshold)
                    .ToList();

                var desni = podaci
                    .Where(p => p.Atributi[atribut].Broj.HasValue && p.Atributi[atribut].Broj.Value > threshold)
                    .ToList();

                double giniLijevi = IzracunajGiniIndeks(lijevi);
                double giniDesni = IzracunajGiniIndeks(desni);

                double giniUkupni = (lijevi.Count * giniLijevi + desni.Count * giniDesni) / podaci.Count;

                if (giniUkupni < najboljiGini)
                {
                    najboljiGini = giniUkupni;
                    najboljiThreshold = threshold;
                }
            }

            return (najboljiGini, najboljiThreshold);
        }

        public static (double gini, double? najboljiThreshold) IzracunajGiniSaThresholdovima(
            List<RedPodatka> podaci,
            string nazivAtributa,
            List<double> thresholds)
        {
            double najboljiGini = double.MaxValue;
            double? najboljiThreshold = null;

            foreach (double t in thresholds)
            {
                var lijevi = podaci
                    .Where(p => p.Atributi[nazivAtributa].Broj.HasValue && p.Atributi[nazivAtributa].Broj.Value <= t)
                    .ToList();

                var desni = podaci
                    .Where(p => p.Atributi[nazivAtributa].Broj.HasValue && p.Atributi[nazivAtributa].Broj.Value > t)
                    .ToList();

                if (lijevi.Count == 0 || desni.Count == 0)
                    continue;

                double giniLijevi = GiniHelper.IzracunajGiniIndeks(lijevi);
                double giniDesni = GiniHelper.IzracunajGiniIndeks(desni);
                double ukupanGini = (lijevi.Count * giniLijevi + desni.Count * giniDesni) / (lijevi.Count + desni.Count);

                if (ukupanGini < najboljiGini)
                {
                    najboljiGini = ukupanGini;
                    najboljiThreshold = t;
                }
            }

            return (najboljiGini, najboljiThreshold);
        }

    }
}
