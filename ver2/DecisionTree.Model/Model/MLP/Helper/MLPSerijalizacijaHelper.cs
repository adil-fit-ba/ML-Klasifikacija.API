using DecisionTree.Model.Model.MLP.MLPMreza;
using OfficeOpenXml.FormulaParsing.LexicalAnalysis;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace DecisionTree.Model.Model.MLP
{
    public class MLPSerijalizacijaHelper
    {

        // Zaključak:
        // Ovaj princip serijalizacije (snimanje mreže: težine + parametri) i rekonstrukcije (ucitavanje i inicijalizacija iste mreže)
        // je isti kao kod velikih modela poput DeepSeek, LLaMA, Mistral itd.
        // Kod DeepSeek-a se također downloadaju trenirane težine (bin fajlovi), a source code za forward/backward (PyTorch) rekonstruiše mrežu.
        // Naš kod (MLPKlasifikator) koristi JSON umjesto bin fajlova, ali rezultat je isti: rekonstruisana mreža sa istim performansama!

        // Dodatna napomena:
        //  Veličina serijalizirane mreže (modela) je mnogo manja veličine originalnog dataset-a!
        //  U tabličnim datasetovima (tipa Excel, CSV), obično je 0.5–1%.
        //  Kod jednostavnih ili jako filtriranih datasetova – može biti ispod 0.1%.
        //  Kod složenih neuronskih mreža za sa mnogo kompleksnih uzoraka (npr. tekst sa stotinama hiljada riječi ili slike u visokoj rezoluciji) ide i iznad 1% (ponekad 3% dataset-a).

        public void SnimiMrezu(MLPKlasifikator mreza, string putanja)
        {
            var dto = new MLPDto
            {
                ParametriMLP = mreza.ParametriMLP,
                KoristiSoftmaxNaIzlazu = mreza.koristiSoftmaxNaIzlazu,
                CiljnaKolona = mreza.CiljnaKolona,
                MLPAtributi = mreza.MLPAtributi,
                Slojevi = mreza.Slojevi.Select(sloj => new LayerDto
                {
                    Neuroni = sloj.Neuroni.Select(neuron => new NeuronDto
                    {
                        Tezine = neuron.Tezine,
                        Bias = neuron.Bias
                        // Delta i Output se ne snimaju

                        /* Delta:
                         *   Delta u neuronima se koristi samo tokom treniranja (u fazi “backpropagation”).
                         *   Nakon treniranja, više se ne koristi za predikciju.
                         *   Nije potrebna za ponovno korištenje mreže – za predikciju koristi samo Tezine i Bias.
                        */
                        /* Output:
                         *   Output se izračuna svaki put kada mreža procesuira neki novi ulaz
                         *   Nema potrebe da se snima, jer se računa "u hodu" (forward pass).
                         */
                    }).ToList()
                }).ToList()
            };

            var json = JsonSerializer.Serialize(dto, new JsonSerializerOptions { WriteIndented = true }); //WriteIndented = true - formatira izlaz da bude čitljiv 
            var direktorij = Path.GetDirectoryName(putanja);
            if (string.IsNullOrEmpty(direktorij) && !Directory.Exists(direktorij))
            {
                Directory.CreateDirectory(direktorij!);
            }

            File.WriteAllText(putanja, json);

            /*
             * ideje za unaprijeđenje:
             *   - verzioniranje (verzijaMreze, datumTreniranja u DTO).
             *   - složenije statističke metrike.
             */
        }

        public MLPKlasifikator UcitajMrezu(string putanja)
        {
            var json = File.ReadAllText(putanja);
            var dto = JsonSerializer.Deserialize<MLPDto>(json);

            var slojevi = new List<Layer>();
            for (int i = 0; i < dto.Slojevi.Count; i++)
            {
                var layerDto = dto.Slojevi[i];

                Func<double, double> aktivacija = i == dto.Slojevi.Count - 1
                    ? AktivacijskeFunkcijeHelper.IzlazniSlojevi.Sigmoid
                    : AktivacijskeFunkcijeHelper.SkriveniSlojevi.ReLU;

                Func<double, double> derivacija = i == dto.Slojevi.Count - 1
                    ? AktivacijskeFunkcijeHelper.IzlazniSlojevi.SigmoidDerivacija
                    : AktivacijskeFunkcijeHelper.SkriveniSlojevi.ReLUDerivacija;

                var layer = new Layer(layerDto.Neuroni.Count, layerDto.Neuroni[0].Tezine.Length, aktivacija);

                for (int j = 0; j < layer.Neuroni.Count; j++)
                {
                    layer.Neuroni[j].Tezine = layerDto.Neuroni[j].Tezine;
                    layer.Neuroni[j].Bias = layerDto.Neuroni[j].Bias;
                }

                slojevi.Add(layer);
            }

            // Rekonstruisati MLPKlasifikator
            return new MLPKlasifikator(
                dto.ParametriMLP, 
                dto.KoristiSoftmaxNaIzlazu, 
                dto.CiljnaKolona, 
                dto.MLPAtributi, 
                slojevi
            );
        }
    }

    // DTO klase
    public class MLPDto
    {
        public MLPKlasifikator.MLPParametri ParametriMLP { get; set; }
        public bool KoristiSoftmaxNaIzlazu { get; set; }
        public AtributMeta CiljnaKolona { get; set; }
        public AtributMeta[] MLPAtributi { get; set; }
        public List<LayerDto> Slojevi { get; set; } = new();
    }

    public class LayerDto
    {
        public List<NeuronDto> Neuroni { get; set; } = new();
    }

    public class NeuronDto
    {
        public double[] Tezine { get; set; }
        public double Bias { get; set; }
    }
}
