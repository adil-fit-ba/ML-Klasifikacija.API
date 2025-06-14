
using System.Text.Json.Serialization;
using DecisionTree.Model.DataSet;

namespace DecisionTree_1;
public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Add services to the container.

        builder.Services.AddControllers()
            .AddJsonOptions(options =>
            {
                options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
            }); 

        // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

        var app = builder.Build();

        // Configure the HTTP request pipeline.
        app.UseDeveloperExceptionPage(); 
        app.UseSwagger();
        app.UseSwaggerUI();

        app.UseAuthorization();


        app.MapControllers();

        app.Run();
    }
}