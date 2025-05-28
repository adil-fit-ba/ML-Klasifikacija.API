using OfficeOpenXml;

namespace DecisionTree.Model.Helper;

public class ExcelHelper
{
    public MojDataSet Ucitaj(string putanja, string? ciljnaVarijabla = null)
    {
        if (!File.Exists(putanja))
            throw new Exception("Excel fajl nije pronađen.");

        ExcelPackage.License.SetNonCommercialOrganization("FIT education");

        using var paket = new ExcelPackage(new FileInfo(putanja));
        var lista = paket.Workbook.Worksheets[0];

        int brojKolona = lista.Dimension.Columns;
        int brojRedova = lista.Dimension.Rows;

        var zaglavlje = new List<string>();
        for (int i = 1; i <= brojKolona; i++)
            zaglavlje.Add(lista.Cells[1, i].Text.Trim());

        var privremeniPodaci = new List<Dictionary<string, string>>();

        for (int i = 2; i <= brojRedova; i++)
        {
            var red = new Dictionary<string, string>();

            for (int j = 1; j <= brojKolona; j++)
            {
                string naziv = zaglavlje[j - 1];
                string vrijednost = lista.Cells[i, j].Text.Trim();

                red[naziv] = vrijednost;
            }

            privremeniPodaci.Add(red);
        }

        var atributiMeta = zaglavlje
            .Select(n => new AtributMeta
            {
                Naziv = n,
                TipAtributa = privremeniPodaci
                    .Where(p => p.ContainsKey(n) && !string.IsNullOrWhiteSpace(p[n]))
                    .All(p => double.TryParse(p[n], out _))
                        ? TipAtributa.Numericki
                        : TipAtributa.Kategoricki
            })
            .ToList();


        var redovi = new List<RedPodatka>();

        for (int i = 0; i < privremeniPodaci.Count; i++)
        {
            var red = new RedPodatka(new());

            foreach (var meta in atributiMeta)
            {
                string vrijednost = privremeniPodaci[i].TryGetValue(meta.Naziv, out var v) ? v : "";

                if (meta.TipAtributa == TipAtributa.Kategoricki)
                {
                    red.Atributi[meta.Naziv] = VrijednostAtributa.NapraviKategorijski(red, vrijednost);
                }
                else if (meta.TipAtributa == TipAtributa.Numericki)
                {
                    if (double.TryParse(vrijednost, out var broj))
                        red.Atributi[meta.Naziv] = VrijednostAtributa.NapraviNumericki(red, broj);
                    else
                        red.Atributi[meta.Naziv] = VrijednostAtributa.NapraviNumericki(red, null);
                }
            }

            redovi.Add(red);
        }

        return new MojDataSet([putanja], redovi, atributiMeta, ciljnaVarijabla); ;
    }

}
