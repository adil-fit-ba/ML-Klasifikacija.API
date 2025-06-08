using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.MLPMreza;

public class Neuron
{
    public double[] Tezine { get; set; }
    public double Bias { get; set; }
    private readonly Func<double, double> aktivacijskaFunkcija;

    public Neuron(int brojUlaza, Func<double, double> aktivacijskaFunkcija)
    {
        var rand = new Random();
        Tezine = new double[brojUlaza];
        for (int i = 0; i < brojUlaza; i++)
        {
            Tezine[i] = rand.NextDouble() - 0.5; // inicijalizacija u rasponu [-0.5, 0.5]
        }
        Bias = rand.NextDouble() - 0.5;

        /*
         * Bias i težine su parametar koji se trenira – prilagođava se u svakom koraku (backpropagation).
         * Njihove početna vrijednost su random broj. Ako bi ove vrijednost bile nula (ili neki isti za sve neurone), svi bi neuroni učili identično - ne bi 
         * imali sposobnost mreže da različito "razmišlja".
         * Random bias + random težine pomažu da mreža razbije simetriju i bolje nauči.
         */

        /*
         * Programer odlučuje koje će se funkcije koristiti za koji sloj - to je dizajn mreže.
         * Alternative:
         *  Skriveni slojevi: 
         *   1 ReLU, standard za CNN (Convolutional Neural Network za slike, video, audio), MLP (Multi-Layer Perceptron za klasifikciju i regresijske probleme.).
         *      + Brzo učenje (manje problema s „vanishing gradient”).
         *      + Pogodna za duboke mreže i velike skupove podataka.
         *      - Ako neuron „uđe u negativnu zonu”, može „umrijeti” jer izlaz ostaje 0 (dead neuron problem).
         *   2 tanh, Hiperbolički tangent, izlaz je između –1 i 1.
         *      za RNN - pamti šta je bilo ranije(kontekst) kod sekvenci - predviđane sljedećih riječi, prepoznavanje govora, predviđanje vremenskih serija.
         *     + dobro za plitke mreže (1 ili 2 skrivena sloja)
         *     + radi sa pozitivnih i negativnim vrijednostima
         *     - vanishing gradient problem - ako je pomak učenja (korekcija) previše mala - učenje nestaje
         *   3 LeakyReLU, kod dubokih mreža (CNN npr GAN), osigurati na neuroni neće umrijeti
         *     - GAN: 
         *          Prvi dio- Generator – pokušava napraviti lažne (ali uvjerljive) podatke.
                    Drugi dio- Discriminator – pokušava prepoznati da li su podaci pravi ili lažni.
         *          Prave realistične slike (npr. „deepfake”). 
         *          Generišu nove slike ljudi koji ne postoje.
         *    
         *   
         *   Izlazni sloj:
         *   1 Sigmoid (binarna klasa)
         *    - samo dvije klase npr. preživo, nije preživio
         *   2 softmax (više klasa)
         *    - više kalsi, npr. prepoznavanje vrste cvijeta (3 vrste).
         *   3 linear (regresija)
         *    - predviđanje cijene kuće (300000, 500000…).
         *   
         * U ovom projektu aktivacijska funkcija je ReLU za skrivene slojeve, a Sigmond za izlazni sloj
         */
        this.aktivacijskaFunkcija = aktivacijskaFunkcija;
    }

    public double Izracunaj(double[] ulazi)
    {
        if (ulazi.Length != Tezine.Length)
        {
            throw new Exception($"Neuron očekuje {Tezine.Length} ulaza, ali je dobio {ulazi.Length} ulaza.");
        }
        double suma = 0.0;
        for (int i = 0; i < ulazi.Length; i++)
        {
            suma += ulazi[i] * Tezine[i];
        }
        suma += Bias;
        return aktivacijskaFunkcija(suma);
    }
}
