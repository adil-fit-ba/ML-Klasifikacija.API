using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.MLPMreza
{
    public class Neuron
    {
        private readonly bool Loguj = true;
        public double[] Tezine { get; set; }
        public double Bias { get; set; }
        public readonly Func<double, double> aktivacijskaFunkcija;


        public Neuron(int brojUlaza, Func<double, double> aktivacijskaFunkcija)
        {
            var rand = new Random();
            Tezine = new double[brojUlaza];
            for (int i = 0; i < brojUlaza; i++)
                Tezine[i] = rand.NextDouble() - 0.5;

            Bias = rand.NextDouble() - 0.5;
            this.aktivacijskaFunkcija = aktivacijskaFunkcija;
        }

        public double Izracunaj(double[] ulazi)
        {
            double suma = 0.0;
            for (int i = 0; i < ulazi.Length; i++)
                suma += ulazi[i] * Tezine[i];

            suma += Bias;
            var Output = aktivacijskaFunkcija(suma);
            return Output;
        }
    }
}
