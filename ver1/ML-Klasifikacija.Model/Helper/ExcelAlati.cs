namespace ML_Klasifikacija.Model.Helper;

using OfficeOpenXml;
public class ExcelAlati
{
    public static MojDataSet Ucitaj(string putanja, string nazivCiljneKolone)
    {
        if (!File.Exists(putanja))

            throw new Exception("Excel fajl nije pronađen.");

        ExcelPackage.License.SetNonCommercialOrganization("FIT education");

        using var paket = new ExcelPackage(new FileInfo(putanja));
        var lista = paket.Workbook.Worksheets[0];

        var zaglavlje = new List<string>();
        int brojKolona = lista.Dimension.Columns;
        int brojRedova = lista.Dimension.Rows;

        for (int i = 1; i <= brojKolona; i++)
            zaglavlje.Add(lista.Cells[1, i].Text.Trim());

        var podaci = new List<RedPodatka>();

        for (int i = 2; i <= brojRedova; i++)
        {
            var atributi = new Dictionary<string, string>();
            string klasa = "";

            for (int j = 1; j <= brojKolona; j++)
            {
                string naziv = zaglavlje[j - 1];
                string vrijednost = lista.Cells[i, j].Text.Trim();

                atributi[naziv] = vrijednost;
            }

            if (!string.IsNullOrWhiteSpace(klasa))
            {
                podaci.Add(new RedPodatka
                {
                    Atributi = atributi,
                    Klasa = klasa
                });
            }
        }

        return new MojDataSet(podaci, nazivCiljneKolone);
    }
}