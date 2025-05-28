using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DecisionTree.Model.Helper;

namespace DecisionTree.Model.DataSet
{
    public static class MojDataSetHelper
    {
        public static List<double> DohvatiBrojeve(List<RedPodatka> podaci, string nazivAtributa, bool sortiraj = false)
        {
            var brojevi = podaci
                .Select(p => p.GetBroj(nazivAtributa))
                .Where(b => b.HasValue)
                .Select(b => b!.Value);

            return sortiraj ? brojevi.OrderBy(x => x).ToList() : brojevi.ToList();
        }

        public static List<string> DohvatiKategorije(List<RedPodatka> podaci, string nazivAtributa, bool sortiraj = false)
        {
            IEnumerable<string> kategorije = podaci
                .Select(p => p.GetText(nazivAtributa))
                .Where(x => !string.IsNullOrEmpty(x))
                .Distinct()
                .Select(b => b!);

            return sortiraj ? kategorije.OrderBy(x => x).ToList() : kategorije.ToList();
        }
    }

}